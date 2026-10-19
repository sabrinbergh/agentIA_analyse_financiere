"""Mesure la RECHERCHE seule (sans modèle de chat) : le bon passage est-il parmi les extraits renvoyés ?

Pour chaque question, on lance la recherche hybride avec le texte de la question, puis on regarde à quel rang
apparaît le premier extrait qui contient la valeur cherchée (colonne valeur_page, sinon valeur_attendue) et si la
page de référence (colonne page_pdf) est présente.

Usage : python diag_recherche.py questions_test_v2.csv        (MISTRAL_API_KEY définie : sert aux embeddings)
        python diag_recherche.py questions_test_v2.csv 40     (profondeur maximale 40, défaut 30)

Limite : l'agent reformule ses requêtes et en lance plusieurs ; ce test utilise la question brute, c'est donc
une approximation de ce que voit l'agent. Il ne dit pas si le modèle LIT correctement, seulement si l'information
arrive jusqu'à lui. Aucun appel à un modèle de chat : coût nul (quelques embeddings).
"""
import csv
import re
import sys

import outils


def motif(valeur):
    v = re.sub(r"\s*%$", "", valeur.strip()).lstrip("+").strip()
    variantes = {v, v.replace(",", "."), v.replace(".", ",")}
    return [re.compile(r"(?<![\d,.])" + r"[\s  ]?".join(re.escape(t) for t in x.split()) + r"(?!\d)") for x in variantes]


def contient(texte, valeurs):
    for val in valeurs.split("|"):
        if val.strip() and any(m.search(texte) for m in motif(val)):
            return True
    return False


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    profondeur = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    with open(sys.argv[1], encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f, delimiter=";"))
    resultats = []
    print(f"{'id':<5}{'niv':<4}{'rang valeur':<13}{'rang page':<11}{'dense':<7}{'bm25':<6}banque / exercice")
    for q in questions:
        cible = (q.get("valeur_page") or q.get("valeur_attendue") or "").strip()
        page = (q.get("page_pdf") or "").strip()
        banque = outils._normaliser_banque(q["banque"]) if q.get("banque") else None
        exercice = int(q["exercice"]) if q.get("exercice") else None
        try:
            hits = outils.rechercher(q["question"], banque, exercice, profondeur)
        except Exception as e:
            print(f"{q['id']:<5} erreur de recherche : {type(e).__name__} : {e}")
            continue
        r_val = next((n for n, h in enumerate(hits, 1) if contient(h["texte"], cible)), None)
        r_page = next((n for n, h in enumerate(hits, 1) if page and str(h.get("page")) == page), None)
        h = hits[r_val - 1] if r_val else None
        resultats.append((q["id"], q.get("niveau", ""), r_val, r_page))
        print(f"{q['id']:<5}{q.get('niveau', ''):<4}{(r_val or '-'):<13}{(r_page or '-'):<11}"
              f"{('oui' if h and h['dans_dense'] else 'non') if h else '-':<7}{('oui' if h and h['dans_bm25'] else 'non') if h else '-':<6}"
              f"{q.get('banque', '')} {q.get('exercice', '')}", flush=True)
    n = len(resultats)
    print(f"\nSur {n} questions (recherche avec la question brute, valeur cherchée dans les extraits) :")
    for seuil in (3, 5, 10, 20, profondeur):
        ok = sum(1 for _, _, r, _ in resultats if r and r <= seuil)
        print(f"  valeur présente dans les {seuil:>2} premiers extraits : {ok}/{n} = {ok / n:.0%}")
    manques = [i for i, _, r, _ in resultats if not r or r > 10]
    print(f"\nQuestions où la valeur n'est PAS dans les 10 premiers extraits (celles que voit l'agent par défaut) : {', '.join(manques) or 'aucune'}")


if __name__ == "__main__":
    main()
