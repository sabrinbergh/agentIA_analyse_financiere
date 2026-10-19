"""Évalue l'agent sur un fichier de questions et calcule le taux de réussite.

Usage :
  python evaluer.py questions_reglage_bnp2024.csv
  python evaluer.py questions_test.csv --modele mistral-large-latest
  python evaluer.py questions_test.csv --limite 5 --pause 2 --web
  python evaluer.py questions_test.csv --pdfs data/rapports      (vérifie aussi les pages citées)

Colonnes lues : id ; question ; valeur_attendue (plusieurs valeurs acceptées séparées par « | ») ; niveau (facultatif).
Colonnes facultatives banque, exercice et valeur_page : servent uniquement à vérifier les pages citées (--pdfs).
valeur_page = valeur à chercher sur la page quand la réponse est un calcul (sinon valeur_attendue).
La banque et l'exercice ne sont PAS donnés à l'agent : ils doivent être dans le texte de la question.

Notation automatique : la valeur attendue doit apparaître dans la réponse. C'est un premier filtre : relisez
les réponses dans le CSV de résultats et corrigez la colonne ok_manuel (1 ou 0) ; le taux final utilise ok_manuel
quand elle est remplie. Le fichier d'évaluation finale ne doit servir qu'à mesurer, jamais à régler l'agent.
"""
import csv
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import agent as ag


def compacter(texte):
    """Retire les espaces (normaux, insécables, fins) entre chiffres : « 48 831 » devient « 48831 »."""
    return re.sub(r"(?<=\d)[   ](?=\d)", "", texte.replace("−", "-"))


def contient_valeur(reponse, attendu):
    """True si une des valeurs attendues (séparées par « | ») figure dans la réponse (virgule ou point décimal)."""
    texte = compacter(reponse)
    for valeur in attendu.split("|"):
        v = compacter(valeur.strip())
        if not v:
            continue
        v = re.sub(r"\s*%$", "", v)                              # « 61,8 % » : on cherche 61,8
        v = v.lstrip("+")                                        # « +7% » : on cherche 7
        variantes = {v, v.replace(",", "."), v.replace(".", ",")}
        for var in variantes:
            if re.search(r"(?<![\d,.])" + re.escape(var) + r"(?!\d)", texte):
                return True
    return False


def pages_citees(reponse):
    """Numéros de page cités dans la réponse (« page 160 », « pages 147 et 148 du PDF »)."""
    pages = []
    for m in re.finditer(r"pages?\s+(\d+(?:\s*(?:,|et)\s*(?:pages?\s*)?\d+)*)", reponse, flags=re.I):
        pages += [int(x) for x in re.findall(r"\d+", m.group(1))]
    return [p for p in dict.fromkeys(pages) if p > 0]


_PDF = {}


def _slug(banque):
    t = unicodedata.normalize("NFD", banque.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").replace("s.a.", "")
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def texte_page(dossier, banque, exercice, n):
    """Texte de la page n du PDF (None si le fichier ou la page est introuvable)."""
    from pypdf import PdfReader
    chemin = Path(dossier) / f"{_slug(banque)}_{exercice}.pdf"
    if str(chemin) not in _PDF:
        _PDF[str(chemin)] = PdfReader(str(chemin)) if chemin.exists() else None
    lecteur = _PDF[str(chemin)]
    if lecteur is None or not 1 <= n <= len(lecteur.pages):
        return None
    try:
        return lecteur.pages[n - 1].extract_text() or ""
    except Exception:
        return None


def valeur_dans_page(texte, attendu):
    """La valeur attendue (alternatives séparées par « | ») figure-t-elle dans le texte brut d'une page ?
    On garde les espaces du tableau (sinon deux colonnes voisines fusionnent) et on tolère un espace dans les milliers."""
    for valeur in attendu.split("|"):
        v = re.sub(r"\s*%$", "", valeur.strip()).lstrip("+").strip()
        if not v:
            continue
        for var in {v, v.replace(",", "."), v.replace(".", ",")}:
            motif = r"(?<![\d,.])" + r"[\s\u00a0\u202f]?".join(re.escape(t) for t in var.split()) + r"(?!\d)"
            if re.search(motif, texte):
                return True
    return False


def citation_ok(reponse, q, dossier):
    """1 si la valeur attendue figure sur au moins une page citée, 0 sinon, « » si on ne peut pas vérifier."""
    pages = pages_citees(reponse)
    if not (dossier and pages and q.get("banque") and q.get("exercice")):
        return ""
    textes = [texte_page(dossier, q["banque"], q["exercice"], n) for n in pages]
    if all(t is None for t in textes):
        return ""
    return int(any(t is not None and valeur_dans_page(t, q.get("valeur_page") or q.get("valeur_attendue", "")) for t in textes))


def lire_csv(chemin):
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)

    def option(nom, defaut=None):
        for i, a in enumerate(sys.argv):
            if a == f"--{nom}" and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return defaut

    questions = lire_csv(args[0])
    if option("limite"):
        questions = questions[: int(option("limite"))]
    pause = float(option("pause", 1))
    modele = option("modele")
    pdfs = option("pdfs")
    agent = ag.construire_agent(avec_web="--web" in sys.argv, modele=modele)
    print(f"Configuration : recherche={os.environ.get('RECHERCHE_MODE', 'hybride')}, "
          f"calculatrice={'non' if os.environ.get('SANS_CALCULATRICE') == '1' else 'oui'}", flush=True)
    # Garde-fou de coût : prix en USD par million de tokens (défaut = Mistral Large 3) et budget maximal du run.
    prix_ent = float(os.environ.get("PRIX_ENTREE", "0.5"))
    prix_sor = float(os.environ.get("PRIX_SORTIE", "1.5"))
    budget = float(option("budget", 2.0))      # USD ; 0 = pas de limite
    tot_ent = tot_sor = 0

    lignes = []
    for n, q in enumerate(questions, start=1):
        r = ag.poser(agent, q["question"])
        ok = contient_valeur(r["reponse"], q.get("valeur_attendue", ""))
        lignes.append({
            "id": q["id"], "niveau": q.get("niveau", ""), "question": q["question"],
            "valeur_attendue": q.get("valeur_attendue", ""), "reponse": r["reponse"].replace("\n", " "),
            "ok_auto": int(ok), "ok_manuel": "",
            "pages_citees": " ".join(map(str, pages_citees(r["reponse"]))),
            "citation_ok": citation_ok(r["reponse"], q, pdfs),
            "outils": " > ".join(a["outil"] for a in r["appels"]), "nb_appels": len(r["appels"]),
            "duree_s": r["duree_s"], "erreur": r["erreur"] or "",
            "via_secours": int(r.get("secours", False)),
            "tokens_entree": r.get("tokens_entree", 0), "tokens_sortie": r.get("tokens_sortie", 0),
        })
        tot_ent += r.get("tokens_entree", 0)
        tot_sor += r.get("tokens_sortie", 0)
        cout = (tot_ent * prix_ent + tot_sor * prix_sor) / 1e6
        if r.get("secours"):
            print("   (réponse donnée par le modèle de secours)", flush=True)
        print(f"[{n}/{len(questions)}] {q['id']} {'OK ' if ok else 'ECHEC'} ({r['duree_s']} s) attendu {q.get('valeur_attendue', '')!r}  | coût cumulé ≈ {cout:.2f} $", flush=True)
        if budget and cout >= budget:
            print(f"\nBUDGET ATTEINT ({cout:.2f} $ >= {budget} $) : arrêt après {n} questions. "
                  "Les résultats partiels sont enregistrés ; relance avec --budget plus grand si tu veux continuer.", flush=True)
            break
        time.sleep(pause)

    etiquette = os.environ.get("ETIQUETTE", "")        # ex. dense, bm25, sans_calc : repère les essais de l'étude comparative
    sortie = Path(f"resultats_eval_{datetime.now():%Y%m%d_%H%M%S}{'_' + etiquette if etiquette else ''}.csv")
    with open(sortie, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(lignes)

    par_niveau = defaultdict(lambda: [0, 0])
    for l in lignes:
        for cle in (l["niveau"] or "tous", "TOTAL"):
            par_niveau[cle][1] += 1
            par_niveau[cle][0] += l["ok_auto"]
    print("\nTaux de réussite (notation automatique) :")
    for cle in sorted(par_niveau, key=lambda c: (c == "TOTAL", c)):
        ok, tot = par_niveau[cle]
        print(f"  {cle:>8} : {ok}/{tot} = {ok / tot:.0%}")
    verifiees = [l for l in lignes if l["citation_ok"] != ""]
    if verifiees:
        bonnes = sum(l["citation_ok"] for l in verifiees)
        print(f"\nPages citées : la valeur figure sur une page citée dans {bonnes}/{len(verifiees)} réponses vérifiables.")
    echecs = [l for l in lignes if not l["ok_auto"]]
    if echecs:
        print("\nÉchecs à relire :")
        for l in echecs:
            print(f"  {l['id']} attendu {l['valeur_attendue']!r} | réponse : {l['reponse'][:160]}")
    print(f"\nTokens : {tot_ent} en entrée, {tot_sor} en sortie, coût estimé ≈ {cout:.2f} $ (prix {prix_ent}/{prix_sor} $ par million).")
    print(f"\nDétail : {sortie}  (relire les réponses, remplir ok_manuel)")


if __name__ == "__main__":
    main()
