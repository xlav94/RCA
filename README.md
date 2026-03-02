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
sbatch sbatch/script.sh ~/scratch/code-snapshots/RCA
```