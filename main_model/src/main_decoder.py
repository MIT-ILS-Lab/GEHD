# Path Config
import logging

from main_model.src.utils.config import load_config, parse_args
from main_model.src.trainer.decoder_trainer import LEHDTrainer

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Send logs to the console (stdout)
        # logging.FileHandler("my_log_file.log"),  # Optionally, write logs to a file
    ],
)

if __name__ == "__main__":
    args = parse_args(path="main_model/config_decoder.yaml")
    config = load_config(args.config)

    trainer = LEHDTrainer(config)
    trainer.run()

# TODO:
#       Add proper logging and wandb, get rid of other logging methods -- check with base trainer
#       Add data generation for mesh data for a sphere,
#       Look into CVRPlib
