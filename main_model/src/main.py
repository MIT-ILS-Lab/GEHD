import logging
import yaml

from main_model.src.utils.config import load_config, parse_args
from main_model.src.trainer.encoder_trainer import GeGnnTrainer
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


TYPE = "decoder"  # Set to "encoder" or "decoder"

if __name__ == "__main__":
    if TYPE == "encoder":
        # Load the encoder config file
        path = "main_model/configs/config_encoder.yaml"
    elif TYPE == "decoder":
        # Load the decoder config file
        path = "main_model/configs/config_decoder.yaml"
    else:
        raise ValueError("Unknown type, please select either 'encoder' or 'decoder'.")

    logging.info(f"Running {TYPE} training.")

    # Load the config file
    args = parse_args(path=path)
    config = load_config(args.config)

    # Pretty-print the config using YAML formatting
    logging.info(
        f"Loaded configuration from {config}:\n{yaml.dump(config, default_flow_style=False, sort_keys=False)}"
    )

    if TYPE == "encoder":
        # Initialize the encoder trainer
        solver = GeGnnTrainer(config)
    elif TYPE == "decoder":
        # Initialize the decoder trainer
        solver = LEHDTrainer(config)
    else:
        raise ValueError("Unknown type, please select either 'encoder' or 'decoder'.")

    solver.run()
