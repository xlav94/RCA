# RCA

### LOGS
Training logs are stored in `tensorboard_logs/<folder>`. To visualize them, run the following command:
```
tensorboard --logdir tensorboard_logs/<folder>
```

## Run in Cluster

### Setup
```bash
mkdir ~/sbatch
mkdir ~/scratch/code-snapshots
git clone git@github.com:<username>/RCA.git
``` 
#### Copy code to cluster
```bash
rsync -a RCA ~/scratch/code-snapshots/ --exclude .git
```
#### Copy sbatch script to cluster
```bash
rsync RCA/script.sh sbatch
```

### Run Sbatch
```bash
sbatch sbatch/script.sh ~/scratch/code-snapshots/RCA
```
#### See output or error logs in `sbatch/logs/`.
```bash
tail -f scratch/logs/slurm-<job_id>-rca.out
tail -f scratch/logs/slurm-<job_id>-rca.error
```