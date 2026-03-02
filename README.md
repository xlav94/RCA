# RCA

## LOGS
Training logs are stored in `src/tensorboard_logs/<folder>`. To visualize them, run the following command:
```
tensorboard --logdir src/tensorboard_logs/<folder>
```

## Run in Cluster

### Setup
```bash
mkdir ~/sbatch
mkdir ~/scratch/code-snapshots
git clone git@github.com:<username>/RCA.git
rsync -a RCA ~/scratch/code-snapshots/ --exclude .git
``` 

### Run Sbatch
```bash
rsync RCA/script.sh sbatch
sbatch sbatch/script.sh ~/scratch/code-snapshots/RCA
```
See output or error logs in `sbatch/logs/`.
```bash
tail -f scratch/logs/slurm-<job_id>-rca.out
tail -f scratch/logs/slurm-<job_id>-rca.error
```