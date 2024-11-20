# IQL-PyTorch for Stretchable Neuromorphic Array

## Setup

1. **Set up the environment**  
   Create the required environment using Conda:
   ```bash
   conda env create -f environment.yml
   ```

2. **Install `D4RL` fork**  
   Clone and install the custom `D4RL` repository:
   ```bash
   git clone https://github.com/water-vapor/D4RL.git
   cd D4RL
   pip install -e .
   ```

## Training

- **Train a single model**  
  To train a model for the `umaze` environment, run:
  ```bash
  python main.py --env-name antmaze-umaze-v2 --act relu
  ```

- **Train multiple models**  
  To train all three models, use:
  ```bash
  ./train_checkpoints.sh
  ```

## Pre-trained Models

Pre-trained models are available in the `ref_ckpts_0411/` directory.

## Evaluation

1. **Evaluate pre-trained models**  
   Assuming the checkpoints are in the `ref_ckpts_0411/` folder, evaluate the ant agents with:
   ```bash
   ./run_computer_exps.sh
   ```

2. **Simulate synapse weights**  
   Generate simulated synapse weights under various conditions by running:
   ```bash
   python generate_simulated_synapse.py
   ```

3. **Run stretchable ant experiments**  
   For experiments with stretchable ant agents, execute:
   ```bash
   ./run_stretch_exps.sh
   ```

