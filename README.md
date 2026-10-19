# Agent d'analyse financière (RAG)

Agent qui répond à des questions chiffrées sur les **documents d'enregistrement universel** de **BNP Paribas**, **Société Générale** et **Crédit Agricole S.A.** (exercices 2024 et 2025), en citant **la page du PDF** comme source.

> Démo en ligne : _lien Streamlit Community Cloud à ajouter_

## Ce qu'il fait

- Recherche **hybride** (sémantique + mots-clés) dans environ 20 000 extraits des six rapports.
- **Calculatrice** dédiée : tout calcul (variation, écart, pourcentage) passe par l'outil, jamais par le modèle.
- Réponse courte : valeur, unité, banque, exercice et page du PDF.
- Refuse de deviner : si la banque n'est pas nommée, il la demande ; si l'information est absente, il le dit.

## Architecture

```
Question
   |
   v
Agent (LangChain 1.x, modèle gpt-6-luna)
   |-- outil rechercher_rapport : filtre banque/exercice -> dense (mistral-embed) + BM25 -> fusion RRF -> 8 à 15 extraits
   |-- outil calculer           : calculatrice sûre (arbre syntaxique, pas d'eval)
   v
Réponse + page du PDF
```

| Élément | Choix |
|---|---|
| Découpage | `RecursiveCharacterTextSplitter` 1200 / 150, page par page, préfixe `[banque \| exercice \| page \| section]` |
| Index | Chroma, 20 338 extraits, embeddings `mistral-embed` |
| Recherche | dense + BM25, fusion par rangs (RRF, K = 60), 50 candidats par méthode |
| Budget | 8 recherches maximum par question |
| Modèle | `gpt-6-luna` (OpenAI), environ 0,0015 $ par question |

## Résultats

Le taux annoncé est mesuré sur un **jeu de test indépendant** (`questions_test_v3.csv`, 40 questions), jamais utilisé pour régler l'agent. Les 30 questions de mise au point (`questions_test_v2.csv`) ont servi à choisir le modèle et à corriger les règles du prompt.

| Passage | Notation automatique | Après relecture manuelle |
|---|---|---|
| 1 | 37/40 (92,5 %) | 39/40 (97,5 %) |
| 2 | 37/40 (92,5 %) | 39/40 (97,5 %) |
| 3 | 38/40 (95 %) | 40/40 (100 %) |
| **Moyenne** | **93,3 %** | **98,3 %** |

Précisions pour lire ces chiffres honnêtement :

- La notation automatique vérifie que la valeur attendue figure dans la réponse. La relecture manuelle compare chaque réponse et la page citée au rapport.
- Deux questions ont été validées à la relecture : **N25** (ratio de levier BNP, l'agent donne 4,48 %, valeur exacte du rapport, alors que l'attendu était l'arrondi 4,5 %) et **N37** (progression du résultat net SG : le rapport affiche lui-même 68,6 %, alors que l'attendu 68,47 % venait de mon propre calcul). Cette validation a été faite après avoir vu les résultats.
- Échantillon de 40 questions : une question vaut 2,5 points, et l'intervalle de confiance à 95 % d'un score de 39/40 va d'environ 87 % à 99 %. Les 3 passages portent sur les mêmes questions : ils mesurent la stabilité du modèle, pas un échantillon plus grand. Écart entre passages : 1 à 2 points.
- Citations : sur les réponses justes, 113 sur 115 citent la bonne page (les deux autres citent la page voisine).
- 37 questions sur 40 sont justes aux trois passages.

### Choix du modèle (sur les 30 questions de mise au point)

| Modèle | Réussite | Coût de 30 questions |
|---|---|---|
| `ministral-14b-2512` (gratuit) | 73 % | 0 $ |
| `gpt-5.4-mini` | 83 % | 0,26 $ |
| `gpt-6-luna` | 87 % | 0,04 $ |
| `gpt-6.1-sol` | 90 % | 0,75 $ |

`gpt-6-luna` offre le meilleur rapport qualité/prix : 3 points de moins que `gpt-6.1-sol` pour un coût 18 fois plus faible. Le choix a été fait sur ces 30 questions, pas sur le jeu de test.

## Limites connues

- Peut confondre le **Groupe Crédit Agricole** et **Crédit Agricole S.A.** (observé sur une question du jeu de test).
- Quand deux bases existent pour un même chiffre, peut retenir la **base de gestion** au lieu du chiffre publié (cas du PNB du pôle CPBS de BNP Paribas, non résolu).
- Peut lire les **comptes sociaux** de la maison mère au lieu des comptes consolidés (cas non résolu sur une question de mise au point).
- Peut donner une valeur plus précise que celle attendue (13,29 % au lieu de 13,3 %) ou une variation calculée avec des chiffres non arrondis.
- Chaque question est traitée **indépendamment** : pas de mémoire de conversation.
- Périmètre fermé : trois banques, deux exercices. Pas de données de marché ni d'autres établissements.
- Il faut toujours **vérifier le chiffre à la page citée**. Outil de démonstration, pas un conseil en investissement.

## Ce qui n'a pas marché

- **Plus de candidats par méthode** (100 au lieu de 50) : rappel dégradé (23/30 au lieu de 24/30 dans le top 10).
- **Réindexer avec des titres de section corrigés et des en-têtes de tableau** (essai sur Crédit Agricole 2024) : gain non concluant (une question entre dans le top 10, une autre sort du top 30). Non retenu. Le code de l'essai est dans `indexer_v2.py`.
- **Modèles Mistral plus puissants** : inaccessibles avec le plan gratuit (erreurs 429 ou 403).

## Installation

```bash
pip install -r requirements.txt
```

1. Placer les six PDF dans `data/rapports/` (téléchargés sur les sites des banques ; non inclus dans le dépôt).
2. Définir les clés dans le terminal (jamais dans le code) :
   - `MISTRAL_API_KEY` : embeddings (offre gratuite suffisante)
   - `OPENAI_API_KEY` : modèle de chat
3. Construire l'index (environ 30 minutes avec l'offre gratuite) : `python indexer_rapports.py`
4. Poser une question : `python agent.py "Quel est le ratio CET1 de Société Générale à fin 2024 ?"`
5. Lancer l'application : `streamlit run app.py`

Évaluation : `python evaluer.py questions_test_v3.csv --pause 1 --budget 0.5 --pdfs data/rapports` (avec `MISTRAL_MODEL=gpt-6-luna`, `PRIX_ENTREE=0.10`, `PRIX_SORTIE=0.50`).

## Structure

| Fichier | Rôle |
|---|---|
| `agent.py` | agent, prompt système, gestion des erreurs et des limites de débit |
| `outils.py` | recherche hybride, calculatrice, recherche web (non utilisée dans la démo) |
| `indexer_rapports.py` | extraction des PDF, découpage, embeddings, index Chroma |
| `evaluer.py` | évaluation : notation automatique, vérification des pages citées, coût |
| `diag_recherche.py` | mesure du rang du bon extrait dans la recherche |
| `app.py` | application Streamlit (limites par session et par jour) |
| `tests/` | 45 tests automatiques (calculatrice, filtres, recherche dans les trois modes, interrupteurs, application), sans appel d'API |
| `.github/workflows/tests.yml` | lance les tests à chaque envoi sur GitHub |
| `publier_index.py` | envoie l'index dans un dépôt Hugging Face privé, pour l'hébergement de la démo |
| `questions_test_v2.csv` | 30 questions de mise au point |
| `questions_test_v3.csv` | 40 questions de test indépendantes |

## Tests et reproductibilité

```bash
pip install pytest
pytest -q
```

Les 45 tests tournent sans clé d'API et sans le vrai index : la recherche est testée sur un petit index temporaire avec un faux modèle d'embedding. Ils couvrent la calculatrice (expressions refusées comprises), la normalisation des banques, les filtres, la recherche en mode hybride, dense seul et BM25 seul, l'interrupteur « sans calculatrice », la détection des pannes et l'application Streamlit (limites par session et par jour, budget épuisé, clés manquantes, téléchargement de l'index). GitHub Actions les relance à chaque envoi.

Interrupteurs pour l'étude comparative (les valeurs par défaut ne changent rien au comportement de l'agent) :

| Variable | Valeurs | Effet |
|---|---|---|
| `RECHERCHE_MODE` | `hybride` (défaut), `dense`, `bm25` | méthode de recherche utilisée |
| `SANS_CALCULATRICE` | `1` | retire l'outil `calculer` et adapte la règle du prompt |
| `ETIQUETTE` | texte libre | ajouté au nom du fichier de résultats de `evaluer.py` |

## Déploiement de la démo

La démo tourne sur **Streamlit Community Cloud** (gratuit, déployé depuis ce dépôt GitHub). L'index, trop volumineux pour GitHub, est stocké dans un dépôt Hugging Face privé et téléchargé au démarrage de l'application.

1. `python publier_index.py PSEUDO/rapports-index` (jeton Hugging Face en écriture dans `HF_TOKEN`).
2. Sur share.streamlit.io : New app, dépôt GitHub, fichier principal `app.py`.
3. Dans les « Secrets » de l'application : `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `INDEX_REPO` (= `PSEUDO/rapports-index`) et `HF_TOKEN` (jeton en lecture seule).

## Sécurité et coûts

- Les clés ne sont jamais dans le dépôt : variables d'environnement en local, « Secrets » sur Streamlit Community Cloud.
- La démo plafonne le nombre de questions par visiteur et par jour (`QUESTIONS_PAR_SESSION`, `QUESTIONS_PAR_JOUR`), et les crédits OpenAI sont prépayés sans rechargement automatique : la dépense ne peut pas dépasser le solde.
