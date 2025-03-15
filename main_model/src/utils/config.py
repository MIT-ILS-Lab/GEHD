import argparse
import yaml


def load_config(config_file):
    """
    Loads a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args(path="main_model/configs/config.yaml"):
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the GeGnn encoder.")
    parser.add_argument(
        "--config",
        type=str,
        default=path,  # Default configuration file
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()
