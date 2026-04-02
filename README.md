# Deep Reinforcement Learning for Portfolio Allocation: A Post-Modern Portfolio Theory Approach

## Getting started
### Installation
```bash
git clone https://github.com/xlav94/RCA.git
cd RCA
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

### Manual Configuration (main.py)
```python
if __name__ == "__main__":
    algo_type = "PPO"  # Change algorithm here ("PPO" or "SAC")
    if is_training:
        seed_everything(train_seed)
        train(algo_type, train_seed=train_seed)
    else:
        # --- UPDATE THESE PATHS FOR TESTING ---
        env = 'models/<path_to_env>'
        model = 'models/<path_to_model>'
        # ----------------------------------------
        
        seed_everything(test_seed)
        test(algo_type, env, model, test_seed)
```
### Run the code
```bash
python -m src.main
```

### See training or testing logs
Logs are stored in `tensorboard_logs/<folder>`. To visualize them, run the following command:
```
tensorboard --logdir tensorboard_logs/<folder>
```


## Configuration Parameters Reference

This table details the hyperparameter settings and environment configurations used for the training and evaluation of the models.

| Category  | Parameter          | Description                                                            | Value    |
|:----------|:-------------------|:-----------------------------------------------------------------------|:---------|
| **Model** | `IS_TRAINING`      | Whether the script is in training or test mode.                        | `BOOL`   |
|           | `HIDDEN_SIZE_LSTM` | Number of features in the hidden state of the LSTM layers.             | `INT`    |
|           | `NUM_LAYERS_LSTM`  | Number of recurrent layers stacked in the LSTM.                        | `INT`    |
|           | `DROPOUT_LSTM`     | Dropout probability applied between LSTM layers.                       | `FLOAT`  |
|           | `USE_CNN`          | Boolean flag to toggle Convolutional Neural Network layers.            | `BOOL`   |
|           | `TOTAL_TIMESTEPS`  | Total number of environment steps to run for training.                 | `INT`    |
|           | `TRAIN_SEED`       | Random seed used to ensure reproducibility during training.            | `INT`    |
|           | `TEST_SEED`        | Random seed used for evaluation/testing environments.                  | `INT`    |
|           | `NUM_CPU`          | Number of CPU cores used for parallel environment execution.           | `INT`    |
|           | `CHECKPOINT`       | Enables periodic saving of model weights.                              | `BOOL`   |
| **PPO**   | `BATCH_SIZE`       | Number of samples per gradient update for PPO.                         | `INT`    |
|           | `LEARNING_RATE`    | Step size for the optimizer during weight updates.                     | `FLOAT`  |
|           | `N_STEPS`          | Steps to run for each environment per update.                          | `INT`    |
|           | `N_EPOCHS`         | Number of times to optimize the surrogate loss per update.             | `INT`    |
| **SAC**   | `BATCH_SIZE`       | Size of the mini-batch sampled from the replay buffer.                 | `INT`    |
|           | `BUFFER_SIZE`      | Maximum capacity of the experience replay buffer.                      | `INT`    |
|           | `GRADIENT_STEPS`   | Number of gradient updates performed after each step.                  | `INT`    |
|           | `LEARNING_RATE`    | Learning rate for the actor and critic networks.                       | `FLOAT`  |
| **ENV**   | `ENV_NAME`         | Name of the specific environment or wrapper used.                      | `STRING` |
|           | `STOCKS`           | List of tickers included in the portfolio universe ('MSFT,GOOG,AMZN'). | `STRING` |
|           | `WINDOW_SIZE`      | The size of the look-back period for historical data.                  | `INT`    |
|           | `OBJECTIVE`        | Target performance metric ('MPT' or 'PMPT').                           | `STRING` |
|           | `NORMALIZE`        | Whether to normalize environment.                                      | `BOOL`   |

## Setup and Run in a Slurm Cluster

### Setup
```bash
mkdir ~/sbatch
mkdir ~/scratch/code-snapshots
git clone git@github.com:<username>/RCA.git
``` 
#### Copy code to cluster
```bash
rsync -a ~/RCA ~/scratch/code-snapshots/ --exclude .git
```
#### Copy sbatch script to cluster
```bash
rsync ~/RCA/script.sh sbatch
```

### Run Sbatch
```bash
sbatch ~/sbatch/script.sh ~/scratch/code-snapshots/RCA
```
#### See output or error logs in `~/scratch/logs/`.
```bash
tail -f ~/scratch/logs/slurm-<job_id>-rca.out
tail -f ~/scratch/logs/slurm-<job_id>-rca.error
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

## Project Structure
```
.
└── RCA/
    ├── data/
    ├── models/
    ├── rapport/
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── env.py
    │   ├── main.py
    │   ├── model.py
    │   ├── portfolio_optimizer.py
    │   └── visualization.py
    ├── tensorboard_logs/
    ├── test/
    │   ├── __init__.py
    │   ├── test_data.py
    │   ├── test_env.py
    │   └── test_model.py
    ├── config.ini
    ├── README.md
    ├── requirements.txt
    └── script.sh