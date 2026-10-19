"""Agent d'analyse financière : LangChain 1.x, Mistral, outils = recherche hybride + calculatrice (+ web).

Usage :
  python agent.py "Quel est le produit net bancaire de BNP Paribas en 2024 ?"
  python agent.py                       # mode conversation
  python agent.py --web "..."           # avec la recherche web
Variables : MISTRAL_API_KEY (obligatoire), MISTRAL_MODEL (défaut mistral-small-latest).
Modèle OpenAI : MISTRAL_MODEL=gpt-6-luna (ou --modele) + OPENAI_API_KEY (pip install langchain-openai) ; embeddings toujours Mistral.
Modèle Gemini : MISTRAL_MODEL=gemini-3.5-flash-lite (ou --modele) + GOOGLE_API_KEY ; les embeddings restent ceux de Mistral.
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent

import outils

SYSTEM_PROMPT = """Tu es un analyste financier. Tu réponds en français, uniquement à partir des rapports annuels \
(documents d'enregistrement universel) de BNP Paribas, Crédit Agricole S.A. et Société Générale, exercices 2024 et 2025, \
que tu consultes avec l'outil rechercher_rapport.

Méthode :
1. Pour toute question chiffrée ou factuelle sur ces banques, appelle d'abord rechercher_rapport en précisant la banque et \
l'exercice dès que la question les donne. Ne réponds jamais de mémoire et n'annonce jamais ce que tu vas faire : appelle \
directement l'outil, puis réponds. Formule la requête comme l'intitulé de la ligne du tableau (« Résultat brut d'exploitation », \
« Coût du risque », « Emplois pondérés »), sans les mots de la question (« quel est », « publié », « exercice », « pour »).
2. Si les extraits ne contiennent pas la réponse, reformule (intitulé exact du tableau, synonymes, mots-clés) et recherche \
à nouveau, trois fois au plus. Si la page contient un tableau « publié / sous-jacent » dont les extraits sont incomplets, \
relance une recherche sur l'intitulé de la ligne seul pour obtenir les deux colonnes. Sinon réponds : « Je n'ai pas trouvé cette information dans les rapports indexés. »
3. Exercice : les chiffres de l'exercice N viennent du rapport de l'exercice N. Le rapport N+1 reprend les chiffres de N en \
comparatif, parfois retraités : privilégie le rapport demandé et signale tout écart. Dans un tableau à plusieurs années, \
vérifie l'en-tête des colonnes avant de choisir la valeur. Les chiffres de l'année précédente (2023 dans le rapport 2024) \
figurent en comparatif : pour une question sur 2023, cherche avec exercice=2024. Ne dis jamais « je ne peux pas » sans avoir cherché.
4. Attention aux pièges : résultat publié ou sous-jacent (colonnes multiples chez Crédit Agricole), groupe ou métier/pôle, \
millions ou milliards d'euros, libellés proches (résultat avant impôt, d'exploitation, net, net part du Groupe). Sauf demande explicite, utilise les comptes consolidés du Groupe, \
jamais les comptes sociaux de la société mère (états financiers de la société « Société Générale S.A. » en fin de rapport). \
Pour une question sur « le groupe », prends \
la ligne consolidée (Groupe, Total), jamais celle d'un pôle ou d'un métier (CIB, CPBS, BCEF...) : relis l'intitulé de la ligne \
et de la colonne. Pour une variation en %, préfère la variation affichée dans le tableau. Quand la question dit « publiée » (ou « publié »), \
prends la colonne publiée / reported, jamais une colonne retraitée, distribuable ou sous-jacente ; dis de quelle base vient \
le chiffre. Donne les montants dans l'unité du \
rapport (millions d'euros) sans les convertir.
5. Tout calcul (variation, ratio, somme, pourcentage) passe par l'outil calculer, jamais de tête.
6. Réponds uniquement pour la banque et l'exercice demandés. Si la question ne nomme aucune banque, ne devine pas et ne \
réponds pas pour les trois : demande laquelle. N'annonce aucun chiffre que tu n'as pas lu dans un extrait.
7. Réponse courte : la valeur avec son unité, la banque, l'exercice et la source, écrite « page X du PDF » (le numéro donné \
dans l'extrait ; jamais « page du rapport »). Si tu hésites entre plusieurs valeurs, dis-le. N'invente rien."""

LOG = Path(os.environ.get("LOG_DIR", "logs")) / "agent.jsonl"


def _construire_llm(nom, max_retries=4):
    """Construit le modèle de chat : Gemini si le nom commence par « gemini », sinon Mistral."""
    from langchain_core.rate_limiters import InMemoryRateLimiter
    max_tokens = int(os.environ.get("MISTRAL_MAX_TOKENS", "1024"))
    if nom.startswith("gemini"):
        # Clé dans GOOGLE_API_KEY ou GEMINI_API_KEY. 0.2 req/s = 12 requêtes/min (limite gratuite : 15).
        from langchain_google_genai import ChatGoogleGenerativeAI
        limiteur = InMemoryRateLimiter(requests_per_second=float(os.environ.get("GEMINI_RPS", "0.2")),
                                       check_every_n_seconds=0.1, max_bucket_size=1)
        cle = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(model=nom, temperature=0, max_retries=max_retries, timeout=120,
                                      rate_limiter=limiteur, max_output_tokens=max_tokens, google_api_key=cle)
    if nom.startswith(("gpt", "o1", "o3", "o4")):
        # OpenAI : clé dans OPENAI_API_KEY. Les modèles à raisonnement refusent « temperature » et comptent le raisonnement
        # dans la limite de sortie : on laisse donc la température par défaut et on prévoit une limite plus large.
        from langchain_openai import ChatOpenAI
        limiteur = InMemoryRateLimiter(requests_per_second=float(os.environ.get("OPENAI_RPS", "2")),
                                       check_every_n_seconds=0.1, max_bucket_size=2)
        options = {}
        if os.environ.get("OPENAI_REASONING_EFFORT"):      # ex. low : réduit le raisonnement, donc le coût
            options["reasoning_effort"] = os.environ["OPENAI_REASONING_EFFORT"]
        return ChatOpenAI(model=nom, max_retries=max_retries, timeout=120, rate_limiter=limiteur,
                          max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4096")), **options)
    from langchain_mistralai import ChatMistralAI
    # Espace les appels au modèle (0.5 req/s par défaut = 1 appel toutes les 2 s) pour éviter les 429.
    limiteur = InMemoryRateLimiter(requests_per_second=float(os.environ.get("MISTRAL_RPS", "0.5")),
                                   check_every_n_seconds=0.1, max_bucket_size=1)
    return ChatMistralAI(model=nom, temperature=0, max_retries=max_retries, timeout=120,
                         rate_limiter=limiteur, max_tokens=max_tokens)


PANNES = ("rate limit", "rate_limited", "resource_exhausted", "unavailable", "high demand", "overloaded",
          "getaddrinfo", "connecterror", "timeout", "connection")


def _est_panne(e):
    txt = f"{type(e).__name__} {e}".lower()
    return bool(re.search(r"\b(429|503)\b", txt)) or any(m in txt for m in PANNES)


def _raison(e):
    return f"{type(e).__name__}: {str(e)[:140]}".replace("\n", " ")


ATTENTES_PRINCIPAL = (10, 30, 60)    # secondes d'attente avant de reposer la question au modèle principal


class AgentAvecSecours:
    """Deux agents : le principal, et un agent de secours utilisé seulement si le principal reste en panne
    (serveur surchargé, quota, connexion) après plusieurs essais espacés. La question est alors reposée
    depuis le début au modèle de secours."""

    def __init__(self, principal, secours, nom_secours, attentes=ATTENTES_PRINCIPAL):
        self.principal, self.secours, self.nom_secours = principal, secours, nom_secours
        self.attentes = tuple(attentes)
        self.derniere_via_secours = False

    def invoke(self, entree, config=None):
        self.derniere_via_secours = False
        for attente in (*self.attentes, None):
            try:
                return self.principal.invoke(entree, config=config)
            except Exception as e:
                if not _est_panne(e):
                    raise
                if attente is not None:
                    print(f"  (modèle principal indisponible [{_raison(e)}], nouvel essai dans {attente} s)",
                          file=sys.stderr, flush=True)
                    time.sleep(attente)
                    continue
                print(f"  (modèle principal toujours indisponible [{_raison(e)}] : bascule sur {self.nom_secours})",
                      file=sys.stderr, flush=True)
        self.derniere_via_secours = True
        return self.secours.invoke(entree, config=config)


def _sans_calculatrice():
    """Interrupteur de l'étude comparative : SANS_CALCULATRICE=1 retire l'outil de calcul."""
    return os.environ.get("SANS_CALCULATRICE", "") == "1"


def _prompt():
    if not _sans_calculatrice():
        return SYSTEM_PROMPT
    regle = "Tout calcul (variation, ratio, somme, pourcentage) passe par l'outil calculer, jamais de tête."
    assert regle in SYSTEM_PROMPT
    return SYSTEM_PROMPT.replace(regle, "Fais soigneusement tout calcul (variation, ratio, somme, pourcentage) et indique la formule.")


def _outils(avec_web):
    liste = [outils.rechercher_rapport] if _sans_calculatrice() else [outils.rechercher_rapport, outils.calculer]
    if avec_web:
        web = outils.creer_outil_web()
        if web is None:
            print("Recherche web indisponible (pip install ddgs).", file=sys.stderr)
        else:
            liste.append(web)
    return liste


def construire_agent(avec_web=False, modele=None, llm=None, secours=None):
    """llm : modèle déjà construit (utile pour les tests) ; sinon Mistral ou Gemini selon le nom.
    secours : modèle de secours (défaut : variable MODELE_SECOURS, ou ministral-14b-2512 si le principal est
    Gemini et MISTRAL_API_KEY est définie ; « aucun » pour le désactiver)."""
    liste = _outils(avec_web)
    if llm is not None:
        return create_agent(llm, liste, system_prompt=_prompt())
    nom = modele or os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    secours = secours or os.environ.get("MODELE_SECOURS")
    if not secours and nom.startswith("gemini") and os.environ.get("MISTRAL_API_KEY"):
        secours = "ministral-14b-2512"
    if not secours or secours.lower() in ("aucun", "none", nom):
        return create_agent(_construire_llm(nom), liste, system_prompt=_prompt())
    # Avec un secours, le principal échoue vite (2 essais internes) pour basculer sans longue attente.
    principal = create_agent(_construire_llm(nom, max_retries=2), liste, system_prompt=_prompt())
    de_secours = create_agent(_construire_llm(secours), liste, system_prompt=_prompt())
    return AgentAvecSecours(principal, de_secours, secours)


def _texte(contenu):
    if isinstance(contenu, str):
        return contenu
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in contenu)


ATTENTES_429 = (10, 30, 60, 90, 120)     # secondes d'attente entre deux essais si Mistral répond 429


def _invoquer(agent, question, limite_etapes):
    """Appelle l'agent ; en cas de 429 (limite de débit), attend puis recommence la question."""
    for i, attente in enumerate((*ATTENTES_429, None)):
        try:
            return agent.invoke({"messages": [{"role": "user", "content": question}]},
                                config={"recursion_limit": limite_etapes})
        except Exception as e:
            txt = f"{type(e).__name__} {e}".lower()
            if "insufficient_quota" in txt or "exceeded your current quota" in txt:
                raise                       # plus de crédit : inutile de réessayer
            limite = "429" in txt or "rate limit" in txt or "rate_limited" in txt or "resource_exhausted" in txt
            reseau = any(m in txt for m in ("getaddrinfo", "connecterror", "timeout", "connection", "503", "unavailable", "high demand", "overloaded"))
            if not (limite or reseau) or attente is None:
                raise
            cause = "limite de débit du fournisseur" if limite else "connexion coupée ou serveur du modèle surchargé"
            print(f"  ({cause} [{_raison(e)}], nouvel essai dans {attente} s, essai {i + 1}/{len(ATTENTES_429)})",
                  file=sys.stderr, flush=True)
            time.sleep(attente)


def _lire(res):
    """Extrait (réponse finale, appels d'outils) du résultat de l'agent."""
    msgs = res["messages"]
    reponse = _texte(msgs[-1].content).strip()
    sorties = {m.tool_call_id: _texte(m.content) for m in msgs if getattr(m, "type", "") == "tool"}
    appels = []
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            appels.append({"outil": tc["name"], "args": tc["args"], "sortie": sorties.get(tc["id"], "")[:400]})
    return reponse, appels


def _tokens(res):
    """Total des tokens (entrée, sortie) facturés pour cette question : somme sur tous les appels au modèle."""
    ent = sor = 0
    for m in res.get("messages", []):
        u = getattr(m, "usage_metadata", None) or {}
        ent += int(u.get("input_tokens", 0) or 0)
        sor += int(u.get("output_tokens", 0) or 0)
    return ent, sor


def poser(agent, question, limite_etapes=25, journal=True):
    """Pose une question. Renvoie {reponse, appels, duree_s, erreur, secours} et ajoute une ligne au journal JSONL."""
    t0 = time.time()
    erreur, reponse, appels, secours, t_ent, t_sor = None, "", [], False, 0, 0
    try:
        outils.nouveau_budget()
        res = _invoquer(agent, question, limite_etapes)
        reponse, appels = _lire(res)
        e1, s1 = _tokens(res)
        t_ent, t_sor = t_ent + e1, t_sor + s1
        secours = getattr(agent, "derniere_via_secours", False)
        if not appels:       # le modèle a répondu sans consulter les rapports : on le relance une fois
            rappel = question + "\n\n(Rappel : appelle rechercher_rapport avant de répondre, sans annoncer ce que tu vas faire.)"
            res = _invoquer(agent, rappel, limite_etapes)
            reponse, appels = _lire(res)
            e2, s2 = _tokens(res)
            t_ent, t_sor = t_ent + e2, t_sor + s2
            secours = secours or getattr(agent, "derniere_via_secours", False)
    except Exception as e:                      # limite d'étapes, API en erreur...
        erreur = f"{type(e).__name__}: {e}"
        reponse = "(Erreur : l'agent n'a pas pu répondre.)"
    resultat = {"question": question, "reponse": reponse, "appels": appels,
                "duree_s": round(time.time() - t0, 1), "erreur": erreur, "secours": secours,
                "tokens_entree": t_ent, "tokens_sortie": t_sor}
    if journal:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), **resultat}, ensure_ascii=False) + "\n")
    return resultat


def afficher(r):
    for a in r["appels"]:
        args = ", ".join(f"{k}={v!r}" for k, v in a["args"].items())
        print(f"  -> {a['outil']}({args})")
    print(f"\n{r['reponse']}\n[{r['duree_s']} s, {len(r['appels'])} appel(s) d'outil]" + (f" ERREUR {r['erreur']}" if r["erreur"] else ""))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    agent = construire_agent(avec_web="--web" in sys.argv)
    if args:
        afficher(poser(agent, " ".join(args)))
    else:
        print("Posez une question (ligne vide pour quitter).")
        while (q := input("> ").strip()):
            afficher(poser(agent, q))
