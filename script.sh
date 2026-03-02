#!/bin/bash
#SBATCH --job-name=rca                           # Job name
#SBATCH --cpus-per-task=4                                   # Ask for 1 CPUs
#SBATCH --mem=8Gb                                           # Ask for 1 GB of RAM
#SBATCH --output=/scratch/logs/slurm-%j-%x.out
#SBATCH --error=/scratch/logs/slurm-%j-%x.error


echo -e "\nCopying code directory into compute node..."
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git"
cd $SLURM_TMPDIR/"$1"

echo -e "\nSetting up Python environment..."
module load python/3.13
export PYTHONUNBUFFERED=1
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo -e "\nCurrently using this Python:"
echo $(which python)
echo "in:"
echo $(pwd)
echo "Virtual environment:"
echo $VIRTUAL_ENV
echo "sbatch file name: $0"
echo -e "\nRunning Python script..."

python main.py

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DEST="~/scratch/out/$TIMESTAMP"

echo -e "\nSaving results back to: $OUTPUT_DEST"
mkdir -p "$OUTPUT_DEST"

# On copie tout sauf l'environnement virtuel pour ne pas gaspiller d'espace
rsync -av --exclude "venv" "$SLURM_TMPDIR/$1/src" "$OUTPUT_DEST"