import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import matplotlib.colors as colors
import csv
import time
import datetime

from utils.points_generator import generate_collocation_points, generate_interior_data
from utils.training import train_inverse_pinn_mixed
from utils.stats import relative_error_analytic


###############################################################################
# 1. Clases y funciones base
###############################################################################

class MLP_u(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function):
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
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function):
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

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for _ in range(order):
        dy = torch.autograd.grad(
            dy, x,
            grad_outputs=torch.ones_like(dy),
            create_graph=True,
            retain_graph=True
        )[0]
    return dy

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)



if __name__ == "__main__":

    set_seed(32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    X_int, X_bnd = generate_collocation_points(N_interior=10000, N_boundary=500)
    X_data, u_data, k_data = generate_interior_data(N_data=10000)

    model_u = MLP_u(
        input_size=2,
        output_size=1,
        hidden_layers= 16,
        hidden_units=64,
        activation_function=nn.Tanh()
    ).to(device)
    model_u.apply(init_weights)

    model_k = MLP_k(
        input_size=2,
        output_size=1,
        hidden_layers= 16,
        hidden_units=64,
        activation_function=nn.Tanh()
    ).to(device)
    model_k.apply(init_weights)

    print( sum(p.numel() for p in model_u.parameters() if p.requires_grad) )
    start_time = time.time()
    
    model_u, model_k, adam_LOSS, LGFGS_LOSS = train_inverse_pinn_mixed(
        model_u, model_k, 
        X_int, X_bnd,
        X_data, u_data, k_data,
        adam_epochs=30000,
        lbfgs_iterations=1500,
        lr_adam=1e-4,
        lr_lbfgs=0.5,
        lambda_bc=1,
        lambda_data=100,
        lambda_pde=1,
        plot_every=5000,
        plot_every_Lbfgs = 500,
    )

    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    
    # Convert to hours:minutes:seconds format
    training_time_formatted = str(datetime.timedelta(seconds=training_time))
    
    print(f"Total training time: {training_time_formatted}")
    conf = "(new_try2)"
    # Optionally save the training time
    with open(f'results/training_times/training_time{conf}.csv', 'w') as file:
        file.write(str(training_time_formatted))

    # Save entire model
    torch.save(model_u, f'results/models/model_u{conf}.pth')
    torch.save(model_u.state_dict(), f'results/models/model_u_weights{conf}.pth')
    torch.save(model_k, f'results/models/model_k{conf}.pth')
    torch.save(model_k.state_dict(), f'results/models/model_k_weights{conf}.pth')


    with open(f'results/losses/weigths_LBFGS{conf}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(LGFGS_LOSS)  # Save as a single row

    with open(f'results/losses/weigths_ADAM{conf}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(adam_LOSS)  # Save as a single row

    kerror = relative_error_analytic(model_u, model_k)
    with open(f'results/errors/relative_error{conf}.csv', 'w') as file:
        file.write(str(kerror))

