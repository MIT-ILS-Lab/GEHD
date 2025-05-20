# Main Model Implementation

This folder contains the implementation of the GEHD (Geometric Encoder and Heavy Decoder) NCO model for solving CVRP in 3D on arbitrary surfaces using 3D meshes.

## Contact

**Damian Gerber**
- Email: [dagerber@mit.edu](mailto:dagerber@mit.edu)
- Alternative Email: [damian@turbokids.com](mailto:damian@turbokids.com)

## Project Structure

```
main_model/
├── configs/                 # Configuration files
├── src/                    # Source code
└── requirements.txt       # Python dependencies
```

### Source Code Structure

```
src/
├── main.py                # Main training script
├── wandbsync.py          # Weights & Biases synchronization
├── architecture/        # Model architecture
│   ├── decoder_architecture.py      # Decoder model implementation
│   └── encoder_architecture.py      # Encoder model implementation
├── data/                 # Data processing and loading
│   ├── decoder_dataloader.py       # Data loading for decoder training
│   ├── encoder_dataloader.py       # Data loading for encoder training
│   ├── generate_meshes.py         # Mesh generation
│   ├── generate_object_data.py    # Object data generation
│   └── generate_problem_data.py   # Problem instance generation
├── notebooks/          # Jupyter notebooks for analysis
├── trainer/             # Training utilities
│   ├── base_trainer.py      # Base training class
│   ├── decoder_trainer.py   # Decoder-specific training
│   └── encoder_trainer.py   # Encoder-specific training
├── utils/              # Utility functions
│   ├── config.py           # Configuration management
│   ├── general_utils.py    # General utility functions
│   ├── lr_scheduler.py     # Learning rate scheduling
│   ├── sampler.py          # Data sampling utilities
│   └── tracker.py          # Training progress tracking
└── visualization/      # Visualization tools
    ├── figures/            # Directory for figure generation for the paper
    ├── visualize_embeddings.py    # Encoder embedding visualization
    ├── visualize_meshes_encoder.py # Mesh visualization from encoder
    ├── visualize_meshes.py        # General mesh visualization
    └── visualize_solutions.py     # Decoder solution visualization
```

## Configuration Files

The project uses several YAML configuration files located in the `configs/` directory:

1. `config_encoder.yaml`: Configuration for the encoder training
   - Encoder-specific parameters
   - Training hyperparameters
   - Data processing settings

2. `config_decoder.yaml`: Configuration for the decoder training
   - Decoder-specific parameters
   - Training hyperparameters

3. `config_problem_generation.yaml`: Configuration for problem data generation
   - Problem instance parameters
   - Generation settings
   - Data format specifications

## Training Process

### 1. Encoder Training

The encoder training process consists of three main steps:

1. **Mesh Generation**
   ```bash
   python main_model/src/data/generate_meshes.py
   ```

2. **Object Generation**
   ```bash
   python main_model/src/data/generate_object_data.py
   ```

3. **Encoder Training**
   ```bash
   python main_model/src/main.py
   ```
   Make sure to set TYPE="encoder".

### 2. Decoder Training

The decoder training process involves two main steps:

1. **Problem Data Generation**
   ```bash
   python main_model/src/data/generate_problem_data.py
   ```

2. **Decoder Training**
   ```bash
   python main_model/src/main.py
   ```
   Make sure to set TYPE="decoder".


### 3. Debugging
When you have troubles with the local imports, please try updating the python path using `export PYTHONPATH=$(pwd):$PYTHONPATH`.

## Visualization

The project includes two main visualization tools to analyze the model's performance and results:

### 1. Embedding Visualization

To visualize the encoder's learned embeddings and distances:

```bash
python main_model/src/visualization/visualize_embeddings.py
```

### 2. Solution Visualization

To visualize the decoder's solutions:

```bash
python main_model/src/visualize_solutions.py
```

## Requirements

Install the required dependencies load the pyproject.toml file in the root:
```bash
pip install . e
```
Or by installing the dependencies in via the requirements file:
```bash
pip install -r requirements.txt
```
