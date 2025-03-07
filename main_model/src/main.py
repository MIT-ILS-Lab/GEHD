import logging
import yaml

from main_model.src.utils.config import load_config, parse_args
from main_model.src.trainer.encoder_trainer import GeGnnSolver

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
    # Load the config file
    args = parse_args()
    config = load_config(args.config)

    # Pretty-print the config using YAML formatting
    logging.info(
        f"Loaded configuration from {config}:\n{yaml.dump(config, default_flow_style=False, sort_keys=False)}"
    )

    solver = GeGnnSolver(config)
    solver.run()
