#!/bin/bash
#SBATCH --job-name=rca
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/home/cedric/scratch/logs/slurm-%j-%x.out
#SBATCH --error=/home/cedric/scratch/logs/slurm-%j-%x.error

echo -e "\nCopying code directory into compute node..."
# On crée le dossier de destination d'abord
mkdir -p "$SLURM_TMPDIR/RCA"

# On copie le CONTENU du dossier passé en argument ($1) vers notre destination
rsync -av "$1/" "$SLURM_TMPDIR/RCA/" --exclude ".git"

# On se déplace et on vérifie où on est
cd "$SLURM_TMPDIR/RCA"
echo "Contenu du dossier actuel ($(pwd)) :"
ls -F  # Ceci affichera la liste des fichiers pour confirmer la présence de main.py

# When using jax[cuda12]
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# module load StdEnv/2023
# module load cudacore/.12.9.1
# module load cudnn

echo -e "\nSetting up Python environment..."
module load python/3.13
export PYTHONUNBUFFERED=1

# On crée le venv
python -m venv venv
source venv/bin/activate

# Installation
python -m pip install --upgrade pip
# On vérifie si le fichier existe avant d'installer
if [ -f "requirement.txt" ]; then
    python -m pip install -r requirement.txt
else
    echo "ERREUR : requirement.txt introuvable dans $(pwd)"
    exit 1
fi

echo -e "\nRunning Python script..."
if [ -f "src/main.py" ]; then
    python -m src.main
else
    echo "ERREUR : src/main introuvable dans $(pwd)"
    exit 1
fi

echo -e "\nJob terminé. Préparation de la sauvegarde..."
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DEST="$HOME/scratch/out/RCA_$TIMESTAMP"
mkdir -p "$OUTPUT_DEST"
echo "Sauvegarde des résultats vers : $OUTPUT_DEST"

rsync -av "$SLURM_TMPDIR/RCA/tensorboard_logs" "$OUTPUT_DEST/"
rsync -av "$SLURM_TMPDIR/RCA/models" "$OUTPUT_DEST/"

echo -e "\nSauvegarde réussie !"
echo "Tu peux voir tes résultats avec : ls -R $OUTPUT_DEST"
