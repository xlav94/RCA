import configparser

from src.model import get_agent
from src.data import DataPipeline
from src.env import CustomEnv

config = configparser.ConfigParser()
config.read('config.ini')

# Configuration parameters for the model
hidden_size_lstm = config.getint('MODEL', 'HIDDEN_SIZE_LSTM')
num_layers_lstm   = config.getint('MODEL', 'NUM_LAYERS_LSTM')
learning_rate     = config.getfloat('MODEL', 'LEARNING_RATE')
n_steps           = config.getint('MODEL', 'N_STEPS')
batch_size        = config.getint('MODEL', 'BATCH_SIZE')
n_epochs          = config.getint('MODEL', 'N_EPOCHS')
total_timesteps   = config.getint('MODEL', 'TOTAL_TIMESTEPS')

# Configuration parameters for the environment
stocks      = config.get('ENV','STOCKS').split(',')
window_size = config.getint('ENV', 'WINDOW_SIZE')
env_name    = config.get('ENV', 'ENV_NAME')

def main():
    df = DataPipeline(tickers=stocks, start_date='2010-01-01', end_date='2024-01-01').get_env_data(feature='Open')
    env = CustomEnv(df, stocks, window_size=window_size, env_name=env_name)

    # Train the model
    model = get_agent(env, hidden_size_lstm=hidden_size_lstm, num_layers_lstm=num_layers_lstm,
                      learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs)

    model.learn(progress_bar=True,
                    total_timesteps=total_timesteps
                    )
    model.save('models/ppo_agent')

if __name__ == "__main__":
    main()
