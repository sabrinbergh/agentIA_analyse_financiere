import hashlib
import re
import sys
from pathlib import Path

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import outils  # noqa: E402

DIM = 64
FILLER = ["Gouvernance conseil d'administration comité des nominations", "Facteurs de risque environnement réglementaire",
          "Responsabilité sociale et environnementale émissions", "Assemblée générale actionnaires dividendes proposés"]
DOCS = [
    ("[BNP Paribas | exercice 2024 | p.144 | Résultats] Produit net bancaire 48 831 46 927 + 4,1 %", "BNP Paribas", 2024, 144),
    ("[BNP Paribas | exercice 2025 | p.76 | Résultats] Produit net bancaire 51 223 48 831 + 4,9 %", "BNP Paribas", 2025, 76),
    ("[Société Générale | exercice 2024 | p.30 | Chiffres clés] Produit net bancaire 26 788 25 104", "Société Générale", 2024, 30),
    ("[Crédit Agricole S.A. | exercice 2024 | p.233 | Résultats] Produit net bancaire 27 181 25 180 + 7,9 %", "Crédit Agricole S.A.", 2024, 233),
    ("[BNP Paribas | exercice 2024 | p.200 | Risque de crédit] Le coût du risque s'établit à 2 999 millions d'euros", "BNP Paribas", 2024, 200),
]


# BM25 donne un score nul ou négatif quand le corpus est minuscule : on ajoute des extraits neutres par banque et exercice.
for _b, _a in sorted({(d[1], d[2]) for d in DOCS}):
    for _j, _t in enumerate(FILLER):
        DOCS.append((f"[{_b} | exercice {_a} | p.{500 + _j} | Divers] {_t}", _b, _a, 500 + _j))


def vecteur(texte):
    """Embedding factice et déterministe (sac de mots haché) : permet de tester la recherche sans API."""
    v = [0.0] * DIM
    for t in re.findall(r"\w+", texte.lower()):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1
    return [x / n for x in v]


class FauxEmbedder:
    def embed_query(self, q):
        return vecteur(q)


@pytest.fixture
def index(tmp_path, monkeypatch):
    """Petit index Chroma temporaire, branché à la place de l'index réel."""
    col = chromadb.PersistentClient(path=str(tmp_path / "idx")).get_or_create_collection(
        "rapports", metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"d{i}" for i in range(len(DOCS))], embeddings=[vecteur(d[0]) for d in DOCS],
            documents=[d[0] for d in DOCS],
            metadatas=[{"banque": d[1], "exercice": d[2], "source": "x", "page": d[3], "section": ""} for d in DOCS])
    monkeypatch.setattr(outils, "_collection", lambda: col)
    monkeypatch.setattr(outils, "_embedder", lambda: FauxEmbedder())
    outils._corpus.cache_clear()
    yield col
    outils._corpus.cache_clear()
