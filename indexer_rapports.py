"""Indexe les 6 rapports dans Chroma (collection "rapports").

Usage :
  python indexer_rapports.py --essai                 # découpe seulement, sans API ni Chroma
  python indexer_rapports.py                         # indexe les 6 rapports
  python indexer_rapports.py bnp_paribas_2025        # indexe un seul rapport
  python indexer_rapports.py --essai --mode=pdf credit_agricole_2024   # force le mode de lecture du PDF
"""
import bisect
import re
import sys
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

TAILLE, OVERLAP = 1200, 150          # variante C
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


# ---------- PDF : un découpage par page, on garde le numéro de page ----------
def chunks_pdf(chemin, banque, exercice, layout=False):
    from pypdf import PdfReader
    if not Path(chemin).exists():
        print(f"  ATTENTION : fichier introuvable : {chemin}")
        return []
    reader = PdfReader(str(chemin))
    print(f"  {chemin.name} : {len(reader.pages)} pages")
    sortie = []
    courant = ""                                   # dernier titre numéroté rencontré (d'une page à l'autre)
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
        for doc in splitter.create_documents([texte]):
            k = bisect.bisect_right(positions, doc.metadata["start_index"]) - 1
            section = titres[k] if k >= 0 else courant
            prefixe = f"[{banque} | exercice {exercice} | p.{num}" + (f" | {section}" if section else "") + "] "
            sortie.append((prefixe + doc.page_content, {"page": num, "section": section}))
        if titres:
            courant = titres[-1]
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
mode_force = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mode=")), None)   # --mode=pdf ou --mode=pdf_layout
essai = "--essai" in sys.argv
docs = [d for d in DOCS if not args or d[0] in args]

if not essai:
    import chromadb
    from langchain_mistralai import MistralAIEmbeddings
    emb = MistralAIEmbeddings(model="mistral-embed")
    client = chromadb.PersistentClient(path="chroma_rapports")
    col = client.get_or_create_collection("rapports", metadata={"hnsw:space": "cosine"})

    def embed_avec_retry(textes, essais=6):
        attente = 10
        for _ in range(essais):
            try:
                return emb.embed_documents(textes)
            except Exception as e:
                print(f"  erreur ({type(e).__name__}), nouvel essai dans {attente} s")
                time.sleep(attente)
                attente *= 2
        raise RuntimeError("Échec après plusieurs essais")

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