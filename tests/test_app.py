"""Tests de l'application Streamlit avec un faux agent (aucun appel d'API, aucun coût)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agent as module_agent

APP = str(Path(__file__).resolve().parent.parent / "app.py")
APPELS = [{"outil": "rechercher_rapport", "args": {"question": "pnb", "banque": "BNP Paribas", "exercice": 2025}, "sortie": "[1] p.76"},
          {"outil": "calculer", "args": {"expression": "1+1"}, "sortie": "2"}]


@pytest.fixture
def env(monkeypatch, tmp_path):
    import streamlit as st
    st.cache_resource.clear()
    (tmp_path / "idx").mkdir()
    (tmp_path / "idx" / "x").write_text("x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MISTRAL_API_KEY", "test")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("QUESTIONS_PAR_SESSION", "2")
    monkeypatch.setenv("QUESTIONS_PAR_JOUR", "3")
    etat = {"mode": "ok"}

    def faux_poser(agent, q, limite_etapes=25, journal=True):
        assert journal is False
        if etat["mode"] == "quota":
            return {"reponse": "", "appels": [], "erreur": "RateLimitError insufficient_quota"}
        if etat["mode"] == "err":
            return {"reponse": "", "appels": [], "erreur": "ValueError boom"}
        return {"reponse": "Réponse **51 223**, page 76 du PDF.", "appels": APPELS, "erreur": None}

    monkeypatch.setattr(module_agent, "poser", faux_poser)
    monkeypatch.setattr(module_agent, "construire_agent", lambda *a, **k: object())
    yield etat
    st.cache_resource.clear()


def _dernier(at):
    """Dernière réponse de l'assistant, lue dans l'état de la session (indépendant de la fin du rafraîchissement de l'affichage)."""
    return at.session_state["messages"][-1]["content"]


def test_demarrage_et_exemples(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and len(at.button) == 4


def test_question_exemple(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert "51 223" in _dernier(at)
    assert at.session_state["posees"] == 1
    assert at.session_state["messages"][-1]["appels"] == APPELS


def test_limite_par_session(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    for q in ("a", "b"):
        at.chat_input[0].set_value(q).run()
    assert at.session_state["posees"] == 2


def test_plafond_global_du_jour(env):
    for _ in range(3):
        AppTest.from_file(APP, default_timeout=30).run().chat_input[0].set_value("q").run()
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("q").run()
    assert "plafond" in _dernier(at)


def test_budget_epuise_ne_consomme_pas_le_plafond(env):
    env["mode"] = "quota"
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("q").run()
    assert "budget de la démo est épuisé" in _dernier(at)


def test_erreur_technique_sans_fuite_de_details(env):
    env["mode"] = "err"
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("q").run()
    assert "erreur technique" in _dernier(at) and "boom" not in _dernier(at)


def test_cles_manquantes(env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "OPENAI_API_KEY" in at.error[0].value


def test_index_absent_sans_depot(env, monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "absent"))
    monkeypatch.delenv("INDEX_REPO", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "introuvable" in at.error[0].value


def test_telechargement_de_l_index(env, monkeypatch, tmp_path):
    import os
    import huggingface_hub
    appel = {}

    def faux_dl(repo_id, repo_type, token, local_dir):
        appel.update(repo_id=repo_id, repo_type=repo_type, token=token)
        os.makedirs(local_dir, exist_ok=True)
        open(os.path.join(local_dir, "x"), "w").write("x")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", faux_dl)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("INDEX_REPO", "pseudo/rapports-index")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and not at.error
    assert appel == {"repo_id": "pseudo/rapports-index", "repo_type": "dataset", "token": "hf_test"}
