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
from utils.savekan import save_kan_checkpoint
# ===== NEW: use pykan =====
from kan import KAN
# ==========================


###############################################################################
# 1. Funciones base
###############################################################################

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

    # ===== REPLACED: MLP_u -> KAN for u =====
    # Hidden config was: hidden_layers=2, hidden_units=4  -> width=[2,4,4,1]
    model_u = KAN(width=[2,12,12,12,12,12,12,1]).to(device)
    # Optional: keep init to be no-op for KAN (only applies to nn.Linear)
    model_u.apply(init_weights)

    # ===== REPLACED: MLP_k -> KAN for k =====
    model_k = KAN(width=[2, 12,12,12,12,12,12, 1]).to(device)
    model_k.apply(init_weights)

    print("Num params " +  str(sum(p.numel() for p in model_u.parameters() if p.requires_grad) ))
    start_time = time.time()

    model_u, model_k, adam_LOSS, LGFGS_LOSS = train_inverse_pinn_mixed(
        model_u, model_k, 
        X_int, X_bnd,
        X_data, u_data, k_data,
        adam_epochs=5000,
        lbfgs_iterations=500,
        lr_adam=1e-3,
        lr_lbfgs=0.1,
        lambda_bc=1,
        lambda_data=100,
        lambda_pde=1,
        plot_every=1000,
        plot_every_lbfgs = 100
    )


    end_time = time.time()
    training_time = end_time - start_time
    training_time_formatted = str(datetime.timedelta(seconds=training_time))
    conf = "_kan(6,12)"

    # Asegura carpetas
    os.makedirs(f"results_kan/models/{conf}", exist_ok=True)
    os.makedirs("results_kan/losses", exist_ok=True)
    os.makedirs("results_kan/errors", exist_ok=True)
    os.makedirs("results_kan/training_times", exist_ok=True)

    # Guarda pérdidas/tiempos (si quieres seguir con CSVs)
    with open(f'results_kan/training_times/training_time{conf}.csv', 'w') as file:
        file.write(str(training_time_formatted))
    with open(f'results_kan/losses/weigths_LBFGS{conf}.csv', 'w', newline='') as file:
        csv.writer(file).writerow(LGFGS_LOSS)
    with open(f'results_kan/losses/weigths_ADAM{conf}.csv', 'w', newline='') as file:
        csv.writer(file).writerow(adam_LOSS)

    # === GUARDAR CORRECTAMENTE (SIN picklear el objeto entero) ===
    # Define la config EXACTA usada para instanciar KAN (ajusta si usaste grid/k, etc.)
    cfg_u = {"width": [2,12,12,12,12,12,12,1]}
    cfg_k = {"width": [2,12,12,12,12,12,12,1]}

    ckpt_path = save_kan_checkpoint(
        model_u=model_u,
        model_k=model_k,
        cfg_u=cfg_u,
        cfg_k=cfg_k,
        losses={"adam": adam_LOSS, "lbfgs": LGFGS_LOSS},
        meta={"training_time": training_time_formatted, "seed": 32, "conf": conf},
        out_dir="results_kan/models"
    )

    # (opcional) además guarda sólo los state_dicts por separado
    torch.save(model_u.state_dict(), f'results_kan/models/model_u_weights{conf}.pth')
    torch.save(model_k.state_dict(), f'results_kan/models/model_k_weights{conf}.pth')

    # Error relativo
    kerror = relative_error_analytic(model_u, model_k)
    with open(f'results_kan/errors/relative_error{conf}.csv', 'w') as file:
        file.write(str(kerror))