#!/bin/bash
#SBATCH --job-name=rca
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
    python src/main.py
else
    echo "ERREUR : src/main.py introuvable dans $(pwd)"
    exit 1
fi

# ... reste du script pour la sauvegarde ...