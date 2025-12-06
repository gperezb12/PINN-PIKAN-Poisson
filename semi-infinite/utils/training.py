
import torch
import yaml
from tqdm import tqdm
from utils.loss_functions import loss_bc, loss_data_k, loss_data_u, loss_pde_inverse
from utils.styled_plots import plot_solution_and_k
from typing import Dict, Any

def load_config(config_path: str = 'config.yml') -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_inverse_pinn_mixed(
    modelU: torch.nn.Module, 
    modelK: torch.nn.Module,
    X_int: torch.Tensor, 
    X_bnd: torch.Tensor, 
    X_data: torch.Tensor, 
    u_data: torch.Tensor, 
    k_data: torch.Tensor,
    adam_epochs: int = 10000,
    lbfgs_iterations: int = 500,
    lr_adam: float = 1e-3,
    lr_lbfgs: float = 0.5,
    lambda_bc: float = 5.0,
    lambda_pde: float = 1.0, 
    lambda_data: float = 1.0,
    plot_every: int = 1000,
    plot_every_lbfgs: int = 500,
) -> tuple[torch.nn.Module, torch.nn.Module, list, list]:
    """Train PINN models for inverse problem using Adam followed by L-BFGS.
    
    Args:
        modelU: Neural network for solution u
        modelK: Neural network for diffusion coefficient k
        X_int: Interior collocation points
        X_bnd: Boundary points
        X_data: Data point coordinates
        u_data: Observed solution values
        k_data: Observed diffusion coefficient values
        adam_epochs: Number of Adam optimization epochs
        lbfgs_iterations: Number of L-BFGS iterations
        lr_adam: Learning rate for Adam
        lr_lbfgs: Learning rate for L-BFGS
        lambda_bc: Weight for boundary condition loss
        lambda_data: Weight for data fitting loss
        plot_every: Plot frequency during Adam phase
        plot_every_lbfgs: Plot frequency during L-BFGS phase
    
    Returns:
        Tuple of (modelU, modelK, adam_loss_history, lbfgs_loss_history)
    """
    # Lists to store losses
    adam_loss_history = []
    lbfgs_loss_history = []
    cfg = load_config()

    optimizer_adam = torch.optim.Adam(list(modelU.parameters()) + list(modelK.parameters()), lr=lr_adam)
    print(">>> FASE 1: Entrenamiento con Adam <<<")
    for epoch in tqdm(range(1, adam_epochs+1), desc="Adam"):
        optimizer_adam.zero_grad()

        pde_loss = loss_pde_inverse(modelU, modelK, X_int)
        data_loss_val_u = loss_data_u(modelU, X_data, u_data)
        data_loss_val_k = loss_data_k(modelK, X_data, k_data)
        bc_loss = loss_bc(modelU, X_bnd)
        total_loss = pde_loss + lambda_data * data_loss_val_u + lambda_data * data_loss_val_k + bc_loss
        total_loss.backward()
        optimizer_adam.step()

        if epoch % plot_every == 0 or epoch == 1 or epoch == adam_epochs:
            adam_loss_history.append(total_loss.item())
            print(f"  [Adam epoch {epoch:5d}] total_loss={total_loss.item():.4e}, "
                  f"pde_loss={pde_loss.item():.4e}, "
                  f"data_loss={data_loss_val_u.item():.4e},"
                  f"data_loss_k={data_loss_val_k.item():.4e}"
                  f"bc_loss={bc_loss.item():.4e}")
            plot_solution_and_k(modelU, modelK, epoch, folder="figs_inverse_gpb",device=cfg['device'])

    print(">>> FASE 2: Entrenamiento con L-BFGS <<<")
    optimizer_lbfgs = torch.optim.LBFGS(
        list(modelU.parameters()) + list(modelK.parameters()),
        lr=lr_lbfgs,
        max_iter=lbfgs_iterations,
        history_size=100
    )

    iteration_lbfgs = [0]
    def closure():
        optimizer_lbfgs.zero_grad()
        pde_loss = loss_pde_inverse(modelU, modelK, X_int)
        data_loss_val_u = loss_data_u(modelU, X_data, u_data)
        data_loss_val_k = loss_data_k(modelK, X_data, k_data)
        loss_bc_data = loss_bc(modelU, X_bnd)

        total_loss = pde_loss + lambda_data * data_loss_val_u + lambda_data * data_loss_val_k + loss_bc_data
        total_loss.backward()
        return total_loss

    for i in tqdm(range(1, lbfgs_iterations+1)):
        iteration_lbfgs[0] += 1
        current_pde = loss_pde_inverse(modelU, modelK, X_int).item()
        current_data_u = loss_data_u(modelU, X_data, u_data).item()
        current_data_k = loss_data_k(modelK, X_data, k_data).item()
        loss_bc_data = loss_bc(modelU, X_bnd).item()
        current_total = current_pde + lambda_data * current_data_u + lambda_data * current_data_k + loss_bc_data
        if (i+1) % plot_every_lbfgs == 0 or (i+1) == lbfgs_iterations:
            lbfgs_loss_history.append(current_total)
            print(f"  [LBFGS iter {i+1:5d}] total_loss={current_total:.4e}, "
                  f"pde_loss={current_pde:.4e}, "
                  f"data_loss_u={current_data_u:.4e},"
                  f"data_loss_k={current_data_k:.4e}"
                  f"bc_loss={loss_bc_data:.4e}")
            plot_solution_and_k(modelU, modelK, adam_epochs + i + 1, folder="figs_inverse_gpb", device=cfg['device'])

    return modelU, modelK, adam_loss_history, lbfgs_loss_history
