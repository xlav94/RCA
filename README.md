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
rsync ~/RCA/script.sh sbatch
```

### Run Sbatch
```bash
sbatch sbatch/script.sh ~/scratch/code-snapshots/RCA
```
#### See output or error logs in `~/scratch/logs/`.
```bash
tail -f scratch/logs/slurm-<job_id>-rca.out
tail -f scratch/logs/slurm-<job_id>-rca.error
```

### Outputs
Outputs from sbatch will be stored in :
```
 ~/scratch/out/RCA_$TIMESTAMP
```
To save the model and tensorboard logs in the repository:
```bash
rsync -av ~/scratch/out/RCA_$TIMESTAMP/models/ppo_agent_$TIMESTAMP ~/RCA/models/
rsync -av ~/scratch/out/RCA_$TIMESTAMP/tensorboard_logs/PPO_<NUMBER> ~/RCA/tensorboard_logs/
```


### Problems with sbatch
```bash
module purge
pkill -u $USER
```