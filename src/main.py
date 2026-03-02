import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

batch_size = config['MODEL']['BATCH_SIZE']