#!/bin/bash
#SBATCH --job-name=rca
#SBATCH --cpus-per-task=4
#SBATCH --mem=8Gb
#SBATCH --output=/home/cedric/scratch/logs/slurm-%j-%x.out   # Chemin absolu recommandé
#SBATCH --error=/home/cedric/scratch/logs/slurm-%j-%x.error

# Assure-toi que le dossier de logs existe AVANT de lancer sbatch
# mkdir -p ~/scratch/logs

echo -e "\nCopying code directory into compute node..."
# On copie le dossier source proprement sans --relative
rsync -av "$1/" "$SLURM_TMPDIR/RCA/" --exclude ".git"
cd "$SLURM_TMPDIR/RCA"

echo -e "\nSetting up Python environment..."
module load python/3.13
export PYTHONUNBUFFERED=1
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo -e "\nRunning Python script..."
python main.py

TIMESTAMP=$(date +%Y%m%d_%H%M)
# Utilisation de $HOME au lieu de ~ pour plus de fiabilité
OUTPUT_DEST="$HOME/scratch/out/$TIMESTAMP"

echo -e "\nSaving results back to: $OUTPUT_DEST"
mkdir -p "$OUTPUT_DEST"

# On sauvegarde tout le dossier RCA (qui contient tes modèles et logs)
# mais on exclut le venv pour économiser de l'espace
rsync -av --exclude "venv" "$SLURM_TMPDIR/RCA/" "$OUTPUT_DEST/"