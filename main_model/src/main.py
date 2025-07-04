"""
This file is used to run the main training script.
"""

import logging
import yaml
import torch

from main_model.src.utils.config import load_config, parse_args
from main_model.src.trainer.encoder_trainer import GeGnnTrainer
from main_model.src.trainer.decoder_trainer import GEHDTrainer

torch.backends.cudnn.benchmark = True

# Set the type of model to run, either "encoder" or "decoder"
TYPE = "decoder"

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Send logs to the console (stdout)
        # logging.FileHandler("my_log_file.log"),  # Optionally, write logs to a file
    ],
)

# Define the models
models = {"encoder": GeGnnTrainer, "decoder": GEHDTrainer}
paths = {
    "encoder": "main_model/configs/config_encoder.yaml",
    "decoder": "main_model/configs/config_decoder.yaml",
}

####################################################################################################
##### Main ########################################################################################
####################################################################################################

if __name__ == "__main__":
    assert (
        TYPE in models.keys()
    ), f"Invalid model type: {TYPE}. Must be one of {models.keys()}"

    # Select the model to run
    model = models[TYPE]
    path = paths[TYPE]

    logging.info(f"Running {TYPE} training.")

    # Load the config file
    args = parse_args(path=path)
    config = load_config(args.config)

    # Pretty-print the config using YAML formatting
    logging.info(
        f"Loaded configuration from {path}:\n{yaml.dump(config, default_flow_style=False, sort_keys=False)}"
    )

    solver = model(config)

    solver.run()
