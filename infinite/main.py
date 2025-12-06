import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import csv
import time
import datetime
from typing import Dict, Any

from utils.points_generator import generate_collocation_points, generate_interior_data
from utils.training import train_inverse_pinn_mixed
from utils.stats import relative_error_analytic
from utils.savekan import save_kan_checkpoint
from kan import KAN


class MLP_u(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: int, hidden_units: int, activation_function: nn.Module):
        super(MLP_u, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_units)
        self.linear_out = nn.Linear(hidden_units, output_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers)])
        self.act = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x


class MLP_k(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: int, hidden_units: int, activation_function: nn.Module):
        super(MLP_k, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_units)
        self.linear_out = nn.Linear(hidden_units, output_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers)])
        self.act = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


def get_activation(name: str) -> nn.Module:
    activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
    return activations.get(name.lower(), nn.Tanh())


def load_config(config_path: str = 'config.yml') -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_pinn_models(cfg: Dict[str, Any], device: torch.device):
    activation = get_activation(cfg['network']['pinn']['activation'])
    model_u = MLP_u(
        cfg['network']['input_size'],
        cfg['network']['output_size'],
        cfg['network']['pinn']['hidden_layers'],
        cfg['network']['pinn']['hidden_units'],
        activation
    ).to(device)
    model_k = MLP_k(
        cfg['network']['input_size'],
        cfg['network']['output_size'],
        cfg['network']['pinn']['hidden_layers'],
        cfg['network']['pinn']['hidden_units'],
        activation
    ).to(device)
    model_u.apply(init_weights)
    model_k.apply(init_weights)
    return model_u, model_k


def create_pikan_models(cfg: Dict[str, Any], device: torch.device):
    model_u = KAN(width=cfg['network']['pikan']['width']).to(device)
    model_k = KAN(width=cfg['network']['pikan']['width']).to(device)
    model_u.apply(init_weights)
    model_k.apply(init_weights)
    return model_u, model_k


def main():
    cfg = load_config()
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_int, X_bnd = generate_collocation_points(
        N_interior=cfg['data']['N_interior'],
        N_boundary=cfg['data']['N_boundary'],
        device=cfg['device'],
    )
    X_data, u_data, k_data = generate_interior_data(N_data=cfg['data']['N_data'], device=cfg['device'])

    if cfg['model_type'] == 'pinn':
        model_u, model_k = create_pinn_models(cfg, device)
        results_dir = 'results'
    else:
        model_u, model_k = create_pikan_models(cfg, device)
        results_dir = 'results_kan'

    print(f"Num params: {sum(p.numel() for p in model_u.parameters() if p.requires_grad)}")
    
    start_time = time.time()
    model_u, model_k, adam_LOSS, LGFGS_LOSS = train_inverse_pinn_mixed(
        model_u, model_k,
        X_int, X_bnd,
        X_data, u_data, k_data,
        adam_epochs=cfg['training']['adam_epochs'],
        lbfgs_iterations=cfg['training']['lbfgs_iterations'],
        lr_adam=cfg['training']['lr_adam'],
        lr_lbfgs=cfg['training']['lr_lbfgs'],
        lambda_bc=cfg['training']['lambda_bc'],
        lambda_pde=cfg['training']['lambda_pde'],
        lambda_data=cfg['training']['lambda_data'],
        plot_every=cfg['training']['plot_every'],
        plot_every_lbfgs=cfg['training']['plot_every_lbfgs']
    )
    
    training_time = str(datetime.timedelta(seconds=time.time() - start_time))
    conf = cfg['output']['conf_suffix']

    os.makedirs(f"{results_dir}/models", exist_ok=True)
    os.makedirs(f"{results_dir}/losses", exist_ok=True)
    os.makedirs(f"{results_dir}/errors", exist_ok=True)
    os.makedirs(f"{results_dir}/training_times", exist_ok=True)

    with open(f'{results_dir}/training_times/training_time{conf}.csv', 'w') as f:
        f.write(training_time)
    with open(f'{results_dir}/losses/weigths_LBFGS{conf}.csv', 'w', newline='') as f:
        csv.writer(f).writerow(LGFGS_LOSS)
    with open(f'{results_dir}/losses/weigths_ADAM{conf}.csv', 'w', newline='') as f:
        csv.writer(f).writerow(adam_LOSS)

    if cfg['model_type'] == 'pikan':
        save_kan_checkpoint(
            model_u, model_k,
            {"width": cfg['network']['pikan']['width']},
            {"width": cfg['network']['pikan']['width']},
            {"adam": adam_LOSS, "lbfgs": LGFGS_LOSS},
            {"training_time": training_time, "seed": cfg['seed'], "conf": conf},
            f"{results_dir}/models"
        )
    
    torch.save(model_u.state_dict(), f'{results_dir}/models/model_u_weights{conf}.pth')
    torch.save(model_k.state_dict(), f'{results_dir}/models/model_k_weights{conf}.pth')

    kerror = relative_error_analytic(model_u, model_k, device=cfg['device'])
    with open(f'{results_dir}/errors/relative_error{conf}.csv', 'w') as f:
        f.write(str(kerror))

    print(f"Training completed in {training_time}")


if __name__ == "__main__":
    main()
