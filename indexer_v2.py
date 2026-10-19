"""Indexe les rapports dans Chroma (collection "rapports"), version 2 : titres de section et en-têtes de colonnes fiables.

Différences avec la version 1 :
  - le titre de section n'est écrit dans le texte de l'extrait que s'il est sur la page ou sur la page précédente
    ou celle d'avant (avant : un titre pouvait être hérité de 5 à 17 pages plus tôt, donc faux) ;
  - on ajoute en tête de l'extrait les lignes d'en-tête de tableau de la page (« (en millions d'euros) 31/12/2024
    31/12/2023 », années en colonnes) quand elles ne sont pas déjà dans l'extrait.
Par défaut l'index est écrit dans « chroma_test » : l'index actuel (chroma_rapports) n'est pas touché.

Usage :
  python indexer_v2.py --essai credit_agricole_2024                # découpe seulement, sans API ni Chroma
  python indexer_v2.py credit_agricole_2024                        # indexe ce rapport dans chroma_test
  python indexer_v2.py --index=chroma_rapports_v2                  # indexe les 6 rapports dans un autre dossier
Pour interroger cet index : $env:CHROMA_DIR = "chroma_test" puis python diag_recherche.py questions_test_v2.csv
"""
import bisect
import re
import sys
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

TAILLE, OVERLAP = 1200, 150          # variante C
DISTANCE_TITRE_MAX = 2               # un titre hérité n'est gardé que s'il date d'au plus 2 pages
TAILLE_LOT = 100
DOSSIER_RAPPORTS = Path("data/rapports")
DOSSIER_TEXTES = Path("data/textes")

# (nom de fichier sans extension, banque, exercice, mode de lecture du PDF)
# "pdf"        : extraction classique
# "pdf_layout" : extraction qui garde les lignes de tableau sur une seule ligne (plus lente)
DOCS = [
    ("bnp_paribas_2024", "BNP Paribas", 2024, "pdf"),
    ("bnp_paribas_2025", "BNP Paribas", 2025, "pdf_layout"),
    ("credit_agricole_2024", "Crédit Agricole S.A.", 2024, "pdf"),
    ("credit_agricole_2025", "Crédit Agricole S.A.", 2025, "pdf"),
    ("societe_generale_2024", "Société Générale", 2024, "pdf"),
    ("societe_generale_2025", "Société Générale", 2025, "pdf"),
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=TAILLE, chunk_overlap=OVERLAP, add_start_index=True
)


TITRE = re.compile(
    r"^\s*((\d{1,2}\.\d{1,2}(\.\d{1,2}){0,2}\s+[A-ZÉÈÀÂÎÔÛÇ][^\d\n]{5,90})|((CHAPITRE|Chapitre|SECTION|Section)\s+\d+\b.{0,80}))\s*$"
)


ANNEE = re.compile(r"\b20[12]\d\b")
MOTS_ENTETE = {"decembre", "janvier", "juin", "septembre", "mars", "en", "millions", "milliards", "euros", "d", "notes", "note",
               "publie", "sous", "jacent", "var", "variation", "exercice", "au", "du", "et", "t1", "t2", "t3", "t4"}


def _sans_accents(t):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def entetes_page(texte, maxi=2):
    """Lignes de la page qui sont uniquement un en-tête de tableau : années ou dates en colonnes, unité (« en millions
    d'euros »), « publié », « sous-jacent ». Une ligne avec d'autres mots (une phrase) est écartée."""
    res = []
    for ligne in texte.split("\n"):
        s = re.sub(r"[\s\u00a0\u202f]+", " ", ligne).strip()
        if not s or len(s) > 100:
            continue
        annees = set(ANNEE.findall(s))
        dates = re.findall(r"\b31/12/20\d\d\b", s)
        unite = re.search(r"en (millions|milliards) d", s, re.I)
        if not (len(annees) >= 2 or len(dates) >= 1 or unite):
            continue
        reste = [m for m in re.findall(r"[a-z0-9]+", _sans_accents(ANNEE.sub(" ", s))) if m not in MOTS_ENTETE and not m.isdigit()]
        if reste:                                   # un mot inconnu : c'est une phrase ou un libellé, pas un en-tête
            continue
        if s not in res:
            res.append(s)
        if len(res) >= maxi:
            break
    return res


# ---------- PDF : un découpage par page, on garde le numéro de page ----------
def chunks_pdf(chemin, banque, exercice, layout=False):
    from pypdf import PdfReader
    if not Path(chemin).exists():
        print(f"  ATTENTION : fichier introuvable : {chemin}")
        return []
    reader = PdfReader(str(chemin))
    print(f"  {chemin.name} : {len(reader.pages)} pages")
    sortie = []
    courant, page_courant = "", 0                  # dernier titre numéroté rencontré et sa page
    for num, page in enumerate(reader.pages, start=1):
        if layout:
            texte = page.extract_text(extraction_mode="layout") or ""
            texte = "\n".join(re.sub(r"[ \t]{2,}", " ", l).strip() for l in texte.split("\n"))
        else:
            texte = page.extract_text() or ""
        if layout and len(texte.strip()) < 40:           # repli : extraction classique si le mode layout ne rend rien
            texte = page.extract_text() or ""
        texte = texte.replace("\x00", "")
        if len(texte.strip()) < 40:
            continue
        if layout and num % 50 == 0:
            print(f"    page {num}/{len(reader.pages)}...", flush=True)
        positions, titres, curseur = [], [], 0
        for ligne in texte.split("\n"):
            if TITRE.match(ligne):
                positions.append(curseur)
                titres.append(ligne.strip()[:120])
            curseur += len(ligne) + 1
        entetes = entetes_page(texte)
        for doc in splitter.create_documents([texte]):
            k = bisect.bisect_right(positions, doc.metadata["start_index"]) - 1
            if k >= 0:
                section = titres[k]
            else:
                section = courant if courant and num - page_courant <= DISTANCE_TITRE_MAX else ""
            prefixe = f"[{banque} | exercice {exercice} | p.{num}" + (f" | {section}" if section else "") + "] "
            corps = doc.page_content
            manquants = [e for e in entetes if e not in corps]
            if manquants:
                corps = "En-têtes de la page : " + " ; ".join(manquants) + "\n" + corps
            sortie.append((prefixe + corps, {"page": num, "section": section}))
        if titres:
            courant, page_courant = titres[-1], num
    return sortie


def chunks_texte(chemin, banque, exercice):
    texte = chemin.read_text(encoding="utf-8")
    positions, titres, curseur = [], [], 0
    for ligne in texte.split("\n"):
        if TITRE.match(ligne) and "   " not in ligne:
            positions.append(curseur)
            titres.append(ligne.strip()[:120])
        curseur += len(ligne) + 1
    sortie = []
    for doc in splitter.create_documents([texte]):
        debut = doc.metadata["start_index"]
        k = bisect.bisect_right(positions, debut) - 1
        section = titres[k] if k >= 0 else ""
        prefixe = f"[{banque} | exercice {exercice}" + (f" | {section}" if section else "") + "] "
        sortie.append((prefixe + doc.page_content, {"page": 0, "section": section}))
    return sortie


def preparer(stem, banque, exercice, source):
    if source in ("pdf", "pdf_layout"):
        return chunks_pdf(DOSSIER_RAPPORTS / f"{stem}.pdf", banque, exercice, layout=(source == "pdf_layout"))
    return chunks_texte(DOSSIER_TEXTES / f"{stem}.txt", banque, exercice)


# ---------- programme principal ----------
args = [a for a in sys.argv[1:] if not a.startswith("--")]
DOSSIER_INDEX = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--index=")), "chroma_test")
mode_force = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mode=")), None)   # --mode=pdf ou --mode=pdf_layout
essai = "--essai" in sys.argv
docs = [d for d in DOCS if not args or d[0] in args]

if not essai:
    import chromadb
    from langchain_mistralai import MistralAIEmbeddings
    emb = MistralAIEmbeddings(model="mistral-embed")
    client = chromadb.PersistentClient(path=DOSSIER_INDEX)
    col = client.get_or_create_collection("rapports", metadata={"hnsw:space": "cosine"})

    def embed_avec_retry(textes, essais=40):
        # réseau coupé ou limite de débit : on attend (de 10 s jusqu'à 60 s entre deux essais) pendant ~30 min
        attente = 10
        for k in range(essais):
            try:
                return emb.embed_documents(textes)
            except Exception as e:
                print(f"  erreur ({type(e).__name__}), essai {k + 1}/{essais}, nouvel essai dans {attente} s", flush=True)
                time.sleep(attente)
                attente = min(attente * 2, 60)
        raise RuntimeError("Échec après plusieurs essais : vérifiez la connexion internet puis relancez la même commande")

total = 0
for stem, banque, exercice, source in docs:
    if mode_force:
        source = mode_force
    t0 = time.time()
    print(f"\n[{stem}] lecture en mode {source}")
    chunks = preparer(stem, banque, exercice, source)
    n = len(chunks)
    total += n
    avec_section = sum(1 for _, m in chunks if m["section"])
    if n == 0:
        print(f"\n{stem} : 0 chunk, aucun texte extrait. Rapport ignoré (voir diagnostic_pdf.py).")
        continue
    print(f"\n{stem} : {n} chunks, {avec_section/n:.0%} avec un titre de section ({time.time()-t0:.0f} s de préparation)")
    if essai:
        for texte, _ in chunks[len(chunks)//2: len(chunks)//2 + 2]:
            print("  exemple :", texte[:160].replace("\n", " ⏎ "))
        avec_entetes = sum(1 for t, _ in chunks if "En-têtes de la page :" in t)
        print(f"  {avec_section / n:.0%} des extraits ont un titre de section, {avec_entetes / n:.0%} ont des en-têtes de colonnes ajoutés")
        for mot in ("Produit net bancaire", "Coefficient d'exploitation", "Résultat net part du Groupe"):
            tableau = next((t for t, _ in chunks if mot in t), None)
            if tableau:
                i = tableau.find(mot)
                print(f"  ligne de tableau ({mot}) :", tableau[i:i + 200].replace("\n", " ⏎ "))
        continue

    deja = len(col.get(where={"source": stem}, include=[])["ids"])
    print(f"  {deja} déjà indexés")
    for debut in range(deja, n, TAILLE_LOT):
        lot = chunks[debut:debut + TAILLE_LOT]
        vecteurs = embed_avec_retry([t for t, _ in lot])
        col.add(
            ids=[f"{stem}-{debut + i}" for i in range(len(lot))],
            embeddings=vecteurs,
            documents=[t for t, _ in lot],
            metadatas=[{"banque": banque, "exercice": exercice, "source": stem,
                        "page": m["page"], "section": m["section"]} for _, m in lot],
        )
        print(f"  {min(debut + TAILLE_LOT, n)}/{n}")

print(f"\nTotal : {total} chunks" + ("" if essai else " (indexation terminée)"))
