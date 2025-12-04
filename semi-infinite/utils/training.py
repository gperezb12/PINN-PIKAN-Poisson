
import torch
from tqdm import tqdm
from utils.loss_functions import loss_bc, loss_data_k, loss_data_u, loss_pde_inverse
from utils.styled_plots import plot_solution_and_k


def train_inverse_pinn_mixed(
    modelU, modelK,
    X_int, X_bnd, 
    X_data, u_data, k_data,
    adam_epochs=10000,
    lbfgs_iterations=500,
    lr_adam=1e-3,
    lr_lbfgs=0.5,
    lambda_bc=5.0, 
    lambda_data=1.0,
    plot_every=1000,
    plot_every_lbfgs = 500,
):
    # Lists to store losses
    adam_loss_history = []
    lbfgs_loss_history = []

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
            plot_solution_and_k(modelU, modelK, epoch, folder="figs_inverse_gpb")

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
        if (i+1) % plot_every_Lbfgs == 0 or (i+1) == lbfgs_iterations:
            lbfgs_loss_history.append(current_total)
            print(f"  [LBFGS iter {i+1:5d}] total_loss={current_total:.4e}, "
                  f"pde_loss={current_pde:.4e}, "
                  f"data_loss_u={current_data_u:.4e},"
                  f"data_loss_k={current_data_k:.4e}"
                  f"bc_loss={loss_bc_data:.4e}")
            plot_solution_and_k(modelU, modelK, adam_epochs + i + 1, folder="figs_inverse_gpb")

    return modelU, modelK, adam_loss_history, lbfgs_loss_history
