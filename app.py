"""Démo Streamlit de l'agent d'analyse financière (BNP Paribas, Société Générale, Crédit Agricole S.A., 2024 et 2025).

Lancement local :  streamlit run app.py
Clés (variables d'environnement, ou « Secrets » sur Streamlit Community Cloud) : OPENAI_API_KEY et MISTRAL_API_KEY.
Index hébergé : INDEX_REPO (dépôt Hugging Face de type dataset, ex. pseudo/rapports-index) et HF_TOKEN (jeton en lecture) ; sinon dossier local.
Réglages facultatifs : QUESTIONS_PAR_SESSION (défaut 10), QUESTIONS_PAR_JOUR (défaut 150), MISTRAL_MODEL (défaut gpt-6-luna).
"""
import os
import threading
from datetime import date

import streamlit as st

# Réglages de la démo : posés avant l'import de agent.py, sans écraser ce que l'on a défini soi-même.
os.environ.setdefault("MISTRAL_MODEL", "gpt-6-luna")
os.environ.setdefault("MODELE_SECOURS", "aucun")

QUESTIONS_PAR_SESSION = int(os.environ.get("QUESTIONS_PAR_SESSION", "10"))
QUESTIONS_PAR_JOUR = int(os.environ.get("QUESTIONS_PAR_JOUR", "150"))
DOSSIER_INDEX = os.environ.get("CHROMA_DIR", "chroma_rapports")

EXEMPLES = [
    "Quel est le résultat net part du Groupe de BNP Paribas pour l'exercice 2025 ?",
    "Quel est le ratio CET1 de Société Générale à fin décembre 2024 ?",
    "De quel pourcentage le produit net bancaire de BNP Paribas a-t-il augmenté entre 2024 et 2025 ?",
    "Quel est le résultat brut d'exploitation publié de Crédit Agricole S.A. pour l'exercice 2024 ?",
]

st.set_page_config(page_title="Agent d'analyse financière", page_icon="📊", layout="centered")


# ---------- ressources partagées entre les visiteurs ----------
@st.cache_resource
def _compteur_global():
    """Plafond de questions par jour pour toute l'application (protège le budget de l'API)."""
    return {"jour": date.today(), "n": 0, "verrou": threading.Lock()}


def _reserver_question():
    """Réserve une question sur le plafond du jour. Renvoie False si le plafond est atteint."""
    c = _compteur_global()
    with c["verrou"]:
        if c["jour"] != date.today():
            c["jour"], c["n"] = date.today(), 0
        if c["n"] >= QUESTIONS_PAR_JOUR:
            return False
        c["n"] += 1
        return True


def _rendre_question():
    c = _compteur_global()
    with c["verrou"]:
        c["n"] = max(0, c["n"] - 1)


@st.cache_resource(show_spinner="Téléchargement de l'index des rapports (première visite, environ une minute)...")
def _assurer_index():
    """Si l'index n'est pas sur le disque (cas de l'hébergement), le télécharge depuis un dépôt Hugging Face
    (variables INDEX_REPO et, si le dépôt est privé, HF_TOKEN). Renvoie un message d'erreur, ou None si tout va bien."""
    if os.path.isdir(DOSSIER_INDEX) and os.listdir(DOSSIER_INDEX):
        return None
    depot = os.environ.get("INDEX_REPO")
    if not depot:
        return f"Index des rapports introuvable (dossier « {DOSSIER_INDEX} » absent et INDEX_REPO non défini)."
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=depot, repo_type="dataset", token=os.environ.get("HF_TOKEN") or None,
                          local_dir=DOSSIER_INDEX)
    except Exception as e:
        return f"Téléchargement de l'index impossible ({type(e).__name__}). Vérifiez INDEX_REPO et HF_TOKEN."
    return None


@st.cache_resource(show_spinner="Chargement de l'agent et de l'index des rapports...")
def _charger_agent():
    import agent as module_agent
    return module_agent, module_agent.construire_agent()


# ---------- affichage ----------
def _afficher_recherches(appels):
    if not appels:
        return
    with st.expander(f"Voir les recherches effectuées ({len(appels)})"):
        for a in appels:
            if a["outil"] == "rechercher_rapport":
                args = a["args"]
                cible = " · ".join(str(args[k]) for k in ("banque", "exercice") if args.get(k))
                st.markdown(f"**Recherche dans les rapports** : « {args.get('question', '')} »" + (f" ({cible})" if cible else ""))
                st.code(a["sortie"], language=None)
            else:
                st.markdown(f"**Calcul** : `{a['args'].get('expression', '')}` = `{a['sortie']}`")


def _message(m):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            _afficher_recherches(m.get("appels"))


def _barre_laterale(restantes):
    with st.sidebar:
        st.header("À propos")
        st.markdown(
            "Agent qui répond **uniquement à partir des documents d'enregistrement universel** de BNP Paribas, "
            "Société Générale et Crédit Agricole S.A. (exercices 2024 et 2025), avec la **page du PDF** comme source.\n\n"
            "Recherche hybride (sémantique + mots-clés) dans environ 20 000 extraits, calculatrice pour tous les calculs, "
            "modèle `gpt-6-luna`."
        )
        st.metric("Questions restantes (cette session)", restantes)
        with st.expander("Fiabilité mesurée"):
            st.markdown(
                "Environ **98 %** de réponses exactes sur 40 questions de test indépendantes "
                "(moyenne de 3 passages, entre 97,5 % et 100 %). Marge d'erreur d'environ ±3 points : l'échantillon est petit.\n\n"
                "Le test n'a pas servi à régler l'agent. Détail et méthode dans le README du projet."
            )
        with st.expander("Limites connues"):
            st.markdown(
                "- Peut confondre le **Groupe Crédit Agricole** et **Crédit Agricole S.A.**\n"
                "- Peut retenir une **base de gestion** au lieu du chiffre publié quand deux bases existent\n"
                "- Peut lire les **comptes sociaux** de la maison mère au lieu des comptes consolidés\n"
                "- Peut arrondir un chiffre ou lire une variation calculée à partir de chiffres non arrondis\n"
                "- Chaque question est traitée **indépendamment** (pas de mémoire de la conversation)\n\n"
                "**Vérifiez toujours le chiffre à la page citée.**"
            )
        st.caption("Outil de démonstration, pas un conseil en investissement.")


# ---------- application ----------
st.title("📊 Agent d'analyse financière")
st.caption("BNP Paribas · Société Générale · Crédit Agricole S.A. · exercices 2024 et 2025")

manquantes = [k for k in ("OPENAI_API_KEY", "MISTRAL_API_KEY") if not os.environ.get(k)]
if manquantes:
    st.error("Configuration incomplète : variable(s) manquante(s) : " + ", ".join(manquantes)
             + ". Sur l'hébergement, ajoutez-les dans les « Secrets » de l'application.")
    st.stop()
erreur_index = _assurer_index()
if erreur_index:
    st.error(erreur_index)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.posees = 0

restantes = max(0, QUESTIONS_PAR_SESSION - st.session_state.posees)
_barre_laterale(restantes)

for m in st.session_state.messages:
    _message(m)

question = None
if not st.session_state.messages:
    st.markdown("**Essayez une question :**")
    for i, exemple in enumerate(EXEMPLES):
        if st.button(exemple, key=f"ex{i}", use_container_width=True):
            question = exemple

saisie = st.chat_input("Posez une question sur les rapports annuels..." if restantes else "Limite de la session atteinte",
                       disabled=not restantes)
question = saisie or question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    _message(st.session_state.messages[-1])
    if not _reserver_question():
        reponse = {"reponse": "La démo a atteint son plafond de questions pour aujourd'hui. Revenez demain, "
                              "ou consultez le README du projet pour la lancer en local.", "appels": [], "erreur": None}
    else:
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les rapports..."):
                try:
                    module_agent, agent = _charger_agent()
                    reponse = module_agent.poser(agent, question, journal=False)
                except Exception as e:             # clé invalide, index illisible...
                    reponse = {"reponse": "", "appels": [], "erreur": f"{type(e).__name__}: {e}"}
        st.session_state.posees += 1
        erreur = (reponse.get("erreur") or "").lower()
        if "insufficient_quota" in erreur or "exceeded your current quota" in erreur or "credit_balance" in erreur:
            _rendre_question()
            reponse["reponse"] = "Le budget de la démo est épuisé pour le moment. Merci de réessayer plus tard."
        elif reponse.get("erreur"):
            reponse["reponse"] = "Une erreur technique est survenue. Merci de reformuler ou de réessayer dans un instant."
    st.session_state.messages.append({"role": "assistant", "content": reponse["reponse"], "appels": reponse["appels"]})
    st.rerun()
