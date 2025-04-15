import yaml

from main_model.src.trainer.decoder_trainer import LEHDTrainer


if __name__ == "__main__":
    ## Load config file
    path = "main_model/configs/config_decoder.yaml"

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    ## Load Trainer class
    # Initialize model and load checkpoint
    trainer = LEHDTrainer(config)
    trainer.visualize()
    trainer.visualize_solutions(keys=[20])
