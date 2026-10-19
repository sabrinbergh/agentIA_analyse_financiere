import agent


def test_prompt_par_defaut_impose_la_calculatrice(monkeypatch):
    monkeypatch.delenv("SANS_CALCULATRICE", raising=False)
    assert "passe par l'outil calculer" in agent._prompt()
    assert [t.name for t in agent._outils(False)] == ["rechercher_rapport", "calculer"]


def test_interrupteur_sans_calculatrice(monkeypatch):
    monkeypatch.setenv("SANS_CALCULATRICE", "1")
    assert "passe par l'outil calculer" not in agent._prompt()
    assert [t.name for t in agent._outils(False)] == ["rechercher_rapport"]


def test_detection_des_pannes_reseau_et_limites():
    assert agent._est_panne(RuntimeError("Error code: 429 - rate limit exceeded"))
    assert agent._est_panne(RuntimeError("503 Service Unavailable"))
    assert agent._est_panne(RuntimeError("getaddrinfo failed"))
    assert not agent._est_panne(ValueError("expression non autorisée"))


def test_lire_extrait_la_reponse_et_les_appels_d_outils():
    class Msg:
        def __init__(self, type_, content, tool_calls=None, tool_call_id=None):
            self.type, self.content, self.tool_calls, self.tool_call_id = type_, content, tool_calls, tool_call_id
    msgs = [Msg("human", "q"),
            Msg("ai", "", tool_calls=[{"id": "1", "name": "calculer", "args": {"expression": "1+1"}}]),
            Msg("tool", "1+1 = 2", tool_call_id="1"),
            Msg("ai", "Le résultat est **2**.")]
    reponse, appels = agent._lire({"messages": msgs})
    assert reponse == "Le résultat est **2**."
    assert appels == [{"outil": "calculer", "args": {"expression": "1+1"}, "sortie": "1+1 = 2"}]
