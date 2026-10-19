"""Outils de l'agent : recherche hybride dans les rapports, calculatrice sûre, recherche web (optionnelle).

L'index (dossier chroma_rapports, collection "rapports") est créé par indexer_rapports.py.
"""
import ast
import contextvars
import math
import operator
import os
import re
import unicodedata
from functools import lru_cache

import chromadb
from langchain.tools import tool
from rank_bm25 import BM25Okapi

DOSSIER_INDEX = os.environ.get("CHROMA_DIR", "chroma_rapports")
K_RRF = 60            # constante de la fusion par rangs (1 / (60 + rang))
N_CANDIDATS = int(os.environ.get("N_CANDIDATS", "50"))   # candidats retenus par chaque méthode avant fusion
K_MIN = int(os.environ.get("K_MIN", "8"))                # nombre minimal d'extraits renvoyés à l'agent
K_MAX = int(os.environ.get("K_MAX", "15"))               # nombre maximal d'extraits renvoyés à l'agent
BANQUES = {"bnp": "BNP Paribas", "agricole": "Crédit Agricole S.A.", "generale": "Société Générale"}


# ---------- accès à l'index ----------
@lru_cache(maxsize=1)
def _collection():
    return chromadb.PersistentClient(path=DOSSIER_INDEX).get_collection("rapports")


@lru_cache(maxsize=1)
def _embedder():
    from langchain_mistralai import MistralAIEmbeddings
    return MistralAIEmbeddings(model="mistral-embed")


def _sans_accents(texte):
    return "".join(c for c in unicodedata.normalize("NFD", texte.lower()) if unicodedata.category(c) != "Mn")


def _tokens(texte):
    return re.findall(r"\w+", _sans_accents(texte))


def _normaliser_banque(valeur):
    v = _sans_accents(valeur or "")
    if not v.strip():
        return None
    if "bnp" in v or "paribas" in v:
        return BANQUES["bnp"]
    if "agricole" in v or v.strip() in ("ca", "casa", "ca sa"):
        return BANQUES["agricole"]
    if "generale" in v or v.strip() in ("sg", "socgen"):
        return BANQUES["generale"]
    raise ValueError(f"banque inconnue : {valeur!r} (choisir parmi : {', '.join(BANQUES.values())})")


def _filtre(banque, exercice):
    conditions = []
    if banque:
        conditions.append({"banque": banque})
    if exercice:
        conditions.append({"exercice": int(exercice)})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@lru_cache(maxsize=16)
def _corpus(banque, exercice):
    """Tous les chunks correspondant au filtre + leur index BM25 (construit une fois, gardé en mémoire)."""
    res = _collection().get(where=_filtre(banque, exercice), include=["documents", "metadatas"])
    ids, docs, metas = res["ids"], res["documents"], res["metadatas"]
    bm25 = BM25Okapi([_tokens(d) for d in docs]) if docs else None
    return ids, docs, metas, bm25


def _prefixe_retire(texte):
    return re.sub(r"^\[[^\]]*\]\s*", "", texte)


def rechercher(question, banque=None, exercice=None, k=10):
    """Recherche hybride : dense (mistral-embed) + BM25, fusionnées par rangs (RRF). Renvoie une liste de dicts."""
    ids, docs, metas, bm25 = _corpus(banque, exercice)
    if not docs:
        return []
    par_id = {i: (d, m) for i, d, m in zip(ids, docs, metas)}

    mode = os.environ.get("RECHERCHE_MODE", "hybride").lower()    # hybride (défaut), dense ou bm25 : étude comparative
    dense, lexical = [], []
    if mode != "bm25":
        vecteur = _embedder().embed_query(question)
        dense = _collection().query(query_embeddings=[vecteur], n_results=min(N_CANDIDATS, len(ids)),
                                    where=_filtre(banque, exercice), include=[])["ids"][0]
    if mode != "dense":
        scores = bm25.get_scores(_tokens(question))
        lexical = [ids[i] for i in sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:N_CANDIDATS] if scores[i] > 0]

    fusion = {}
    for classement in (dense, lexical):
        for rang, i in enumerate(classement, start=1):
            fusion[i] = fusion.get(i, 0.0) + 1.0 / (K_RRF + rang)
    meilleurs = sorted(fusion, key=fusion.get, reverse=True)[:k]
    return [{"id": i, "texte": par_id[i][0], **par_id[i][1], "score": fusion[i],
             "dans_dense": i in dense, "dans_bm25": i in lexical} for i in meilleurs]


# ---------- budget de recherches par question (évite les boucles de 30 appels) ----------
_BUDGET = contextvars.ContextVar("budget_recherches", default=None)


def nouveau_budget(max_appels=None):
    """À appeler au début de chaque question : fixe le nombre maximal d'appels à rechercher_rapport."""
    _BUDGET.set({"n": 0, "max": int(max_appels or os.environ.get("MAX_RECHERCHES", "8"))})


# ---------- outil 1 : recherche dans les rapports ----------
@tool
def rechercher_rapport(question: str, banque: str = "", exercice: int = 0, k: int = 10) -> str:
    """Cherche dans les rapports annuels (documents d'enregistrement universel) de BNP Paribas,
    Crédit Agricole S.A. et Société Générale, exercices 2024 et 2025.

    Args:
        question: ce que l'on cherche, avec les mots exacts du rapport si possible (ex. "Produit net bancaire 2024",
            "coût du risque", "ratio CET1").
        banque: "BNP Paribas", "Crédit Agricole S.A." ou "Société Générale". Laisser vide pour chercher dans toutes.
        exercice: 2024 ou 2025. Laisser 0 pour chercher dans les deux.
        k: nombre d'extraits renvoyés (10 par défaut, entre 8 et 15).

    Renvoie des extraits numérotés avec banque, exercice, page du PDF et section.
    """
    budget = _BUDGET.get()
    if budget is not None:
        budget["n"] += 1
        if budget["n"] > budget["max"]:
            return ("Limite de recherches atteinte pour cette question : réponds avec les extraits déjà obtenus, "
                    "ou dis que l'information n'a pas été trouvée.")
    try:
        b = _normaliser_banque(banque)
    except ValueError as e:
        return f"Erreur : {e}"
    if exercice not in (0, 2024, 2025):
        return ("Erreur : exercice doit être 2024, 2025 ou 0 (les deux). Les chiffres de 2023 figurent en comparatif "
                "dans le rapport 2024 : cherche avec exercice=2024.")
    k = max(K_MIN, min(int(k), max(K_MIN, K_MAX)))   # jamais moins de K_MIN extraits : un petit k fait rater les tableaux
    try:
        hits = rechercher(question, b, exercice or None, k)
    except Exception as e:                      # index absent, API en erreur...
        return f"Erreur de recherche : {type(e).__name__} : {e}"
    if not hits:
        return "Aucun extrait trouvé pour ce filtre."
    blocs = []
    for n, h in enumerate(hits, start=1):
        page = h.get("page") or "?"
        section = h.get("section") or "section inconnue"
        blocs.append(f"[{n}] {h['banque']} | exercice {h['exercice']} | page {page} du PDF | {section}\n{_prefixe_retire(h['texte'])}")
    return "\n\n".join(blocs)


# ---------- outil 2 : calculatrice sûre ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FONCTIONS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}


def _eval(noeud):
    if isinstance(noeud, ast.Expression):
        return _eval(noeud.body)
    if isinstance(noeud, ast.Constant) and isinstance(noeud.value, (int, float)) and not isinstance(noeud.value, bool):
        return noeud.value
    if isinstance(noeud, ast.BinOp):
        if isinstance(noeud.op, ast.Pow):
            base, exposant = _eval(noeud.left), _eval(noeud.right)
            if abs(exposant) > 100:
                raise ValueError("exposant trop grand")
            return base ** exposant
        if type(noeud.op) in _OPS:
            return _OPS[type(noeud.op)](_eval(noeud.left), _eval(noeud.right))
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in _OPS:
        return _OPS[type(noeud.op)](_eval(noeud.operand))
    if isinstance(noeud, ast.Call) and isinstance(noeud.func, ast.Name) and noeud.func.id in _FONCTIONS and not noeud.keywords:
        return _FONCTIONS[noeud.func.id](*[_eval(a) for a in noeud.args])
    raise ValueError("expression non autorisée")


def calculer_expression(expression):
    e = expression.strip()
    e = e.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-").replace("^", "**")
    e = re.sub(r"(?<=\d)[   ](?=\d{3}(?!\d))", "", e)       # 2 704 -> 2704
    e = re.sub(r"(?<=\d),(?=\d)", ".", e)                               # 2704,9 -> 2704.9
    return _eval(ast.parse(e, mode="eval"))


@tool
def calculer(expression: str) -> str:
    """Calculatrice exacte. À utiliser pour TOUT calcul (variation, ratio, somme, pourcentage), jamais de tête.

    Args:
        expression: opération arithmétique avec + - * / ** et parenthèses, par exemple
            "(48831 - 46927) / 46927 * 100". Les fonctions round(x, n), abs, min, max, sqrt sont permises.
            Écrire les nombres sans espace (48831) ; la virgule ou le point sont acceptés comme séparateur décimal.
            Pour un pourcentage, multiplier par 100 (le signe % n'est pas accepté).
    """
    try:
        valeur = calculer_expression(expression)
    except ZeroDivisionError:
        return "Erreur : division par zéro."
    except Exception as e:
        return f"Erreur : {e}. Reformulez avec des nombres, + - * / ** et des parenthèses."
    if isinstance(valeur, float):
        valeur = float(f"{valeur:.10g}")
    return f"{expression} = {valeur}"


# ---------- outil 3 (optionnel) : recherche web ----------
def creer_outil_web():
    """Renvoie l'outil de recherche web, ou None si la bibliothèque ddgs n'est pas installée."""
    try:
        from ddgs import DDGS
    except ImportError:
        return None

    @tool
    def rechercher_web(requete: str) -> str:
        """Recherche sur le web. À utiliser seulement pour une information absente des rapports (actualité,
        définition d'un terme, information postérieure à 2025). Les chiffres des rapports se cherchent
        d'abord avec rechercher_rapport.

        Args:
            requete: la recherche, en français de préférence.
        """
        try:
            resultats = list(DDGS().text(requete, region="fr-fr", max_results=5))
        except Exception as e:
            return f"Erreur de recherche web : {type(e).__name__}"
        if not resultats:
            return "Aucun résultat web."
        return "\n\n".join(f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}" for i, r in enumerate(resultats, 1))

    return rechercher_web
