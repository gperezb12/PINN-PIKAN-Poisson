# PINN-PIKAN-Poisson

Physics-Informed Neural Networks (PINNs) and Physics-Informed Kolmogorov-Arnold Networks (PIKANs) for solving inverse Poisson problems in unbounded domains.

## Overview

This project compares two approaches for solving inverse Poisson problems:

- **PINN**: Traditional Multi-Layer Perceptrons with physics-based loss functions
- **PIKAN**: Kolmogorov-Arnold Networks as an alternative architecture

Both methods learn the solution field `u(x,y)` and coefficient `k(x,y)` simultaneously from interior collocation points, boundary conditions, and sparse measurement data.

The repository contains implementations for two domain types:
- `infinite/` - Infinite domain problems
- `semi-infinite/` - Semi-infinite domain problems

## Installation

```bash
git clone git@github.com:gperezb12/PINN-PIKAN-Poisson.git
cd PINN-PIKAN-Poisson
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch 2.4.0, pykan 0.2.8, NumPy, Matplotlib, and PyYAML.

## Running the Code

Navigate to either `infinite/` or `semi-infinite/` directory and run:

```bash
python main.py
```

Results are saved in `results/` for PINN or `results_kan/` for PIKAN by default. The user can also edit this value as desired, in the config.yml file.

## Configuration

Edit `config.yml` in the domain folder you're working with.

### Model Type

```yaml
model_type: 'pinn'    # Use 'pinn' for MLP or 'pikan' for KAN
```

- **`'pinn'`**: Traditional neural network (faster, more stable)
- **`'pikan'`**: Kolmogorov-Arnold Network (experimental, potentially more expressive)

### Random Seed

```yaml
seed: 32              # Any integer for reproducibility
```

Change this to get different random initializations.

### Device

```yaml
device: 'cpu'         # Use 'cpu' or 'cuda'
```

- `'cpu'` - Runs on CPU (slower but always available)
- `'cuda'` - Runs on GPU (requires NVIDIA GPU with CUDA)

### Data Generation

```yaml
data:
  N_interior: 10000   # Number of interior collocation points
  N_boundary: 500     # Number of boundary points (Applicable for semi-infinite domain only)
  N_data: 10000       # Number of measurement data points
```

Increase these values for better accuracy at the cost of training speed. Typical ranges are 1,000 to 50,000.

### Network Architecture

For PINN (MLP):

```yaml
network:
  pinn:
    hidden_layers: 16      # Number of hidden layers
    hidden_units: 32       # Neurons per layer
    activation: 'tanh'     # Activation function: 'tanh', 'relu', 'sigmoid'
```

More layers/units improve approximation but slow down training. The `'tanh'` activation is standard for PINNs.

For PIKAN (KAN):

```yaml
network:
  pikan:
    width: [2, 6, 6, 6, 1]  # Network width: [input, hidden..., output]
    grid: 5                 # Grid size for KAN
    k: 3                    # Polynomial order for KAN
```

The first number in `width` is input dimension (2 for 2D problems), last is output dimension (1), and middle numbers are hidden layer widths. For example, `[2, 8, 8, 1]` creates 2 hidden layers with 8 units each.

The `grid` parameter controls the number of grid points for the spline basis functions (typical range: 3-10). The `k` parameter sets the polynomial order for the splines (typical range: 2-5).

### Training Parameters

```yaml
training:
  adam_epochs: 15000         # Adam optimizer epochs
  lbfgs_iterations: 1500     # L-BFGS optimizer iterations
  lr_adam: 1.0e-4            # Adam learning rate
  lr_lbfgs: 0.5              # L-BFGS learning rate
  lambda_bc: 1               # Boundary condition loss weight
  lambda_data: 100           # Data loss weight
  lambda_pde: 1              # PDE residual loss weight
  plot_every: 5              # Plot frequency during Adam
  plot_every_lbfgs: 500      # Plot frequency during L-BFGS
```

| Parameter | Purpose | Typical Range |
|-----------|---------|---------------|
| `adam_epochs` | Pre-training duration | 5,000-20,000 |
| `lbfgs_iterations` | Fine-tuning duration | 500-5,000 |
| `lr_adam` | Adam step size | 1e-5 to 1e-3 |
| `lambda_bc` | Boundary condition weight | 1-10 |
| `lambda_data` | Data fitting weight | 10-1000 |
| `lambda_pde` | Physics weight | 1-10 |

Adjust lambda weights if one loss term dominates. Increase if that constraint isn't being satisfied, decrease if it's over-emphasized.

### Output

```yaml
output:
  results_dir: 'results'     # Output folder: 'results' or 'results_kan'
  conf_suffix: '(2,4)'       # Identifier for this run
```

Set `results_dir` to match your model type (`'results'` for PINN, `'results_kan'` for PIKAN). Use `conf_suffix` to identify different runs.

## Results

After training, check:

```
results/  (or results_kan/)
├── models/              # Trained model weights (.pth files)
├── losses/              # Loss history (CSV files)
├── errors/              # Relative error metrics
├── training_times/      # Training duration
└── figs/               # Plots (if generated)
```

Key files:
- `model_u_weights{conf}.pth` - Solution field network
- `model_k_weights{conf}.pth` - Coefficient network
- `relative_error{conf}.csv` - Final accuracy metric
- `training_time{conf}.csv` - Total training time



## Workflow

1. Edit `config.yml` with desired parameters
2. Run `python main.py`
3. Monitor console output for loss values
4. Check results in `results/` or `results_kan/`
5. Compare runs by changing `conf_suffix`

## Notes

- Start with the default PINN configuration
- Use `'tanh'` activation for smooth solutions
- GPU recommended for N_interior > 10,000
- PIKAN requires more tuning than PINN

## Project Structure

```
PINN-PIKAN-Poisson/
├── infinite/
│   ├── config.yml
│   ├── main.py
│   └── utils/
├── semi-infinite/
│   ├── config.yml
│   ├── main.py
│   └── utils/
├── requirements.txt
└── README.md
```

## Troubleshooting

- **CUDA out of memory**: Reduce `N_interior`, `N_boundary`, or `N_data`
- **Loss not decreasing**: Lower `lr_adam`, increase `adam_epochs`, or adjust lambda weights
- **Poor accuracy**: Increase network size, training epochs, or collocation points
- **Training too slow**: Use GPU, reduce data points, or use smaller network

## License

See LICENSE file for details.
