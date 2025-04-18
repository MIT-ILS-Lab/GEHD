import logging
import yaml
import torch

from main_model.src.utils.config import load_config, parse_args
from main_model.src.trainer.encoder_trainer import GeGnnTrainer
from main_model.src.trainer.decoder_trainer import LEHDTrainer

torch.backends.cudnn.benchmark = True

# Set the type of model to run, either "encoder" or "decoder"
TYPE = "decoder"


####################################################################################################
##### Main ########################################################################################
####################################################################################################

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
models = {"encoder": GeGnnTrainer, "decoder": LEHDTrainer}
paths = {
    "encoder": "main_model/configs/config_encoder.yaml",
    "decoder": "main_model/configs/config_decoder.yaml",
}

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


# TODO:

# 7. Things to pay attention to:
#   - 1. The feasibility loss, if to addit and where to get it from
#   - 4. The mlp / encoder architecture

# 8. Research what training set to choose (real world, terrain, drones, etc., previous studies at best, something to compare to)

# 9. How to train encoder, which output embedding dimension to use

# 10. Fix logging of testing gap if needed
# 11. Add a gap testing for different sizes, 200, 1000
# 12. Add ability to load model from checkpoint
# 13. Add a stopping criterion to not run the whole sequences

# Compare both models for size 200 and 1000 to see which to choose (maybe one is way better than the other)
# Write function for that, mean and variance for 100, 200, 1000, and plot them
# Visualize the results on the mesh
