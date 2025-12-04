import torch
import numpy as np


def relative_error_analytic(model_u, model_k, n_points=200, device='cuda'):
    """
    Computes the relative percentage error between model predictions and ground truth for u and k.
    
    Args:
        model_u (torch.nn.Module): Trained model predicting u(x, y).
        model_k (torch.nn.Module): Trained model predicting k(x, y).
        alpha (float): Parameter in the analytic solution of u.
        beta (float): Parameter in the analytic solution of u.
        epsilon (float): Parameter in the analytic solution of k.
        n_points (int): Number of grid points in each dimension.
        device (str): 'cpu' or 'cuda'.

    Returns:
        Tuple of relative percentage errors for u and k: (rel_error_u, rel_error_k)
    """
    alpha = 0.5
    beta = 10
    epsilon = 1
    # Crear la malla
    x = np.linspace(-5, 5, n_points)
    y = np.linspace(-5, 5, n_points)
    X, Y = np.meshgrid(x, y, indexing='ij')
    XY_np = np.stack([X.flatten(), Y.flatten()], axis=1)

    # Convertir a tensor para usar en los modelos
    XY_tensor = torch.tensor(XY_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model_u(XY_tensor).cpu().numpy().reshape(n_points, n_points)
        k_pred = model_k(XY_tensor).cpu().numpy().reshape(n_points, n_points)

    # Soluciones analíticas
    u_true = np.exp(-alpha * (X**2 + Y**2)) * np.cos(beta * Y)
    k_true = 1 + 2 / (1 + np.exp(-Y / epsilon))

    # Evitar división por cero
    u_true_safe = np.where(np.abs(u_true) < 1e-8, 1e-8, u_true)
    k_true_safe = np.where(np.abs(k_true) < 1e-8, 1e-8, k_true)

    # Error relativo porcentual promedio
    rel_err_u = np.mean(np.abs((u_pred - u_true) / u_true_safe)) * 100
    rel_err_k = np.mean(np.abs((k_pred - k_true) / k_true_safe)) * 100

    return rel_err_k




