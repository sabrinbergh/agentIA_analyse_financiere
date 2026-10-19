import pytest

import outils


# ---------- calculatrice ----------
@pytest.mark.parametrize("expression, attendu", [
    ("(51223 - 48831) / 48831 * 100", 4.8991),
    ("2 704,9 - 100", 2604.9),            # espaces et virgule décimale
    ("round((51 223-48 831)/48 831*100, 2)", 4.9),
    ("2**10", 1024),
    ("−5 + 3", -2),                       # signe moins typographique
])
def test_calculer_ok(expression, attendu):
    assert outils.calculer_expression(expression) == pytest.approx(attendu, abs=1e-3)


@pytest.mark.parametrize("expression", ["__import__('os')", "open('x')", "2**1000", "a + 1", "[1, 2]", "True + 1"])
def test_calculer_refuse_les_expressions_dangereuses(expression):
    with pytest.raises(Exception):
        outils.calculer_expression(expression)


def test_outil_calculer_messages_d_erreur():
    assert "division par zéro" in outils.calculer.invoke({"expression": "1/0"})
    assert outils.calculer.invoke({"expression": "5 %"}).startswith("Erreur")
    assert outils.calculer.invoke({"expression": "100 * 1.5"}) == "100 * 1.5 = 150.0"


# ---------- normalisation des banques et filtres ----------
@pytest.mark.parametrize("saisie, attendu", [
    ("bnp", "BNP Paribas"), ("BNP Paribas", "BNP Paribas"), ("Société Générale", "Société Générale"),
    ("SG", "Société Générale"), ("crédit agricole", "Crédit Agricole S.A."), ("CA", "Crédit Agricole S.A."),
    ("", None), (None, None),
])
def test_normaliser_banque(saisie, attendu):
    assert outils._normaliser_banque(saisie) == attendu


def test_normaliser_banque_inconnue():
    with pytest.raises(ValueError):
        outils._normaliser_banque("Natixis")


def test_filtre():
    assert outils._filtre(None, None) is None
    assert outils._filtre("BNP Paribas", None) == {"banque": "BNP Paribas"}
    assert outils._filtre("BNP Paribas", 2024) == {"$and": [{"banque": "BNP Paribas"}, {"exercice": 2024}]}


# ---------- recherche ----------
def test_recherche_respecte_le_filtre(index):
    hits = outils.rechercher("produit net bancaire", "BNP Paribas", 2025, k=5)
    assert hits[0]["page"] == 76
    assert all(h["banque"] == "BNP Paribas" and h["exercice"] == 2025 for h in hits)


def test_recherche_hybride_trouve_la_bonne_page(index):
    hits = outils.rechercher("produit net bancaire", "Crédit Agricole S.A.", 2024, k=3)
    assert hits[0]["page"] == 233 and hits[0]["dans_dense"] and hits[0]["dans_bm25"]


def test_recherche_sans_resultat(index):
    assert outils.rechercher("produit net bancaire", "Société Générale", 2025) == []


@pytest.mark.parametrize("mode, dense, bm25", [("hybride", True, True), ("dense", True, False), ("bm25", False, True)])
def test_modes_de_recherche_de_l_etude_comparative(index, monkeypatch, mode, dense, bm25):
    monkeypatch.setenv("RECHERCHE_MODE", mode)
    hits = outils.rechercher("coût du risque millions", "BNP Paribas", 2024, k=5)
    assert hits and hits[0]["page"] == 200
    assert any(h["dans_dense"] for h in hits) is dense
    assert any(h["dans_bm25"] for h in hits) is bm25


def test_mode_par_defaut_est_hybride(index, monkeypatch):
    monkeypatch.delenv("RECHERCHE_MODE", raising=False)
    hits = outils.rechercher("produit net bancaire", "BNP Paribas", 2024, k=5)
    assert any(h["dans_dense"] for h in hits) and any(h["dans_bm25"] for h in hits)


def test_outil_rechercher_rapport_formate_les_sources(index):
    sortie = outils.rechercher_rapport.invoke({"question": "produit net bancaire", "banque": "bnp", "exercice": 2025})
    assert "page 76 du PDF" in sortie and "51 223" in sortie
    assert "[Résultats]" not in sortie            # le préfixe technique est retiré du texte


def test_outil_rechercher_rapport_banque_inconnue(index):
    sortie = outils.rechercher_rapport.invoke({"question": "x", "banque": "Natixis"})
    assert "inconnue" in sortie.lower() or "erreur" in sortie.lower()


def test_budget_de_recherches(index):
    outils.nouveau_budget(max_appels=2)
    appels = [outils.rechercher_rapport.invoke({"question": "produit net bancaire", "banque": "bnp"}) for _ in range(3)]
    assert "page" in appels[0] and "page" in appels[1]
    assert "page" not in appels[2].split("\n")[0].lower() or "budget" in appels[2].lower() or "limite" in appels[2].lower()
