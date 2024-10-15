# CoGIR
ControlNet-Guided Image Refinement

## Table of Contents

1. [Getting Started](#getting-started)
2. [Training a Model](#training-a-model)
3. [Extending the Configuration](#extending-the-configuration)
4. [Creating a Custom Config](#creating-a-custom-config)
5. [Developing the Codebase](#developing-the-codebase)
6. [Running Experiments and Tracking with Weights & Biases](#running-experiments-and-tracking-with-weights--biases)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA (for GPU support)
- [Weights & Biases](https://wandb.ai) account (optional for experiment tracking)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CoGIR.git
   cd CoGIR
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training a Model

You can train a model using the provided configuration files. The training process is managed by PyTorch Lightning and Hydra for configuration management.

To start training with the default settings, run:
```bash
python train.py
```

This will use the configuration specified in `config/config.yaml` and will log training results to Weights & Biases if you have it configured.

To run with custom configurations, you can extend or modify any part of the config. For example, you can specify a custom dataset location and batch size:
```bash
python train.py data.dataset.location=/path/to/dataset data.batch_size=8
```

---

## Extending the Configuration

Hydra allows you to compose and extend configurations. The default configuration (`config/config.yaml`) includes references to multiple components like the model, optimizer, and logger.

To extend or modify the configuration, you can create a new YAML file or pass specific overrides through the command line.

### Example: Custom Loss Function

If you want to use a different loss function, you can modify the criterion as follows in a YAML file:
```yaml
criterion:
  _target_: torch.nn.CrossEntropyLoss
```

Then, run the training with:
```bash
python train.py --config-name custom_loss.yaml
```

Alternatively, you can override a specific part of the config directly through the command line:
```bash
python train.py criterion._target_=torch.nn.CrossEntropyLoss
```

---

## Creating a Custom Config

We highly recommend creating custom configuration files rather than modifying the default YAML files (`config/config.yaml`). This keeps your project organized and allows for easy switching between different environments or experiment settings.

To create a custom config file, you can use the following structure:

1. **Create your custom YAML config file**:  
   For example, `jeremy-local.yaml` is a custom configuration:
   ```yaml
   defaults:
     - config
     - _self_

   data:
     dataset:
       location: /media/disk3/unsplash-lite_data
     batch_size: 4
   train:
     gpus: [0, 1]
   logger:
     notes: "Hydra/Lightning refactor"
   criterion:
     _target_: criterion.lpips_loss.LPIPSLoss
     use_l1: true
   ```

2. **Use the custom config**:  
   Run the training command with your custom config:
   ```bash
   python train.py --config-name jeremy-local.yaml
   ```

This allows you to easily switch between different setups (e.g., local vs remote machine) without editing the default configs.

---

## Developing the Codebase

1. **Project Structure**:
    - `train.py`: The entry point for model training.
    - `model/`: Contains the UNet model architecture.
    - `data/`: Contains data loaders and dataset utilities.
    - `config/`: Configuration files managed by Hydra.
    - `tests/`: Unit tests to ensure the code works as expected.

2. **Running Unit Tests**:
   We use `pytest` for unit testing. To run the tests:
    ```bash
    pytest tests/
    ```

3. **Adding New Features**:
   To add new models, datasets, or loss functions, create the necessary modules in `model/`, `data/`, or `criterion/` respectively, and update the corresponding YAML configuration file.

---

## Running Experiments and Tracking with Weights & Biases

CoGIR is set up to integrate seamlessly with [Weights & Biases](https://wandb.ai) for experiment tracking.

### Setup

1. Ensure that you have a Weights & Biases account and have logged in:
   ```bash
   wandb login
   ```

2. In the configuration file (`config/logger/wandb.yaml`), modify the project and entity name:
   ```yaml
   project: CoGIR
   entity: your-wandb-entity
   ```

### Tracking Metrics

During training, metrics like loss and validation accuracy will be logged automatically. You can also log custom images and other data to W&B from the training script.

To visualize input-output pairs, for example, the `train.py` script logs images every few steps:
```python
self.logger.log_image(key="train/input_output_target", images=[wandb.Image(x) for x in input_output_target])
```

### Running a W&B Tracked Experiment

To run a new experiment with W&B tracking enabled, simply run:
```bash
python train.py
```

Make sure you have modified the W&B configuration to suit your project. All experiment data will be saved and viewable in your W&B dashboard.

---