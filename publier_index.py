"""Envoie l'index (dossier chroma_rapports) dans un dépôt Hugging Face PRIVÉ de type dataset, pour que l'application hébergée
puisse le télécharger au démarrage.

Usage (PowerShell) :
  $env:HF_TOKEN = Read-Host "Jeton HF (écriture)" -AsSecureString | ConvertFrom-SecureString -AsPlainText   # PowerShell 7
  ou, en PowerShell 5 :  $s = Read-Host "Jeton HF" -AsSecureString; $env:HF_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))
  python publier_index.py sab3333/rapports-index
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    sys.exit("Usage : python publier_index.py PSEUDO/NOM_DU_DEPOT")
depot = sys.argv[1]
dossier = Path(os.environ.get("CHROMA_DIR", "chroma_rapports"))
if not dossier.is_dir():
    sys.exit(f"Dossier introuvable : {dossier}")
if not os.environ.get("HF_TOKEN"):
    sys.exit("Définissez d'abord HF_TOKEN (jeton Hugging Face avec droit d'écriture).")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(depot, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=str(dossier), repo_id=depot, repo_type="dataset",
                  ignore_patterns=[".cache/**"], commit_message="Index des rapports")
print(f"Index envoyé dans le dépôt privé {depot}.")
