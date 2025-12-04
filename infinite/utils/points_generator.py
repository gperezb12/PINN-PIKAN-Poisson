
import numpy as np
import torch



def generate_interior_data(N_data=100, device='cuda'):
    x_data = np.random.normal(loc=0.0, scale=2, size=(N_data, 1))
    y_data = np.random.normal(loc=0.0, scale=2, size=(N_data, 1))
    X_data_np = np.hstack((x_data, y_data))
    
    # Solución analítica
    alpha = 0.5
    beta = 10
    epsilon = 1
    
    u_data_np = np.exp(-alpha * (x_data**2 + y_data**2)) * np.cos(beta * y_data)
    k_true = 1 + 2 / (1 + np.exp(-y_data / epsilon))
    
    # Agregar ruido del 5%
    u_noise = 0 * np.random.randn(*u_data_np.shape) * np.abs(u_data_np)
    k_noise = 0 * np.random.randn(*k_true.shape) * np.abs(k_true)
    
    u_data_np_noisy = u_data_np 
    k_data_np_noisy = k_true 

    # Convertir a tensores
    X_data = torch.tensor(X_data_np, dtype=torch.float32, requires_grad=True, device=device)
    k_data = torch.tensor(k_data_np_noisy, dtype=torch.float32, requires_grad=True, device=device)
    u_data = torch.tensor(u_data_np_noisy, dtype=torch.float32, device=device)
    
    return X_data, u_data, k_data

def generate_collocation_points(N_interior=2000, N_boundary=200, std_dev=2.5, device='cuda'):
    """
    Genera puntos de entrenamiento: interior (para el residual de la PDE)
    y frontera (para la condiciÃ³n de frontera u=0), en el dominio [-3,3]^2.

    Args:
        N_interior (int): NÃºmero de puntos interiores.
        N_boundary (int): NÃºmero de puntos en la frontera (y=0).

    Returns:
        (X_int, X_bnd) en formato (torch.Tensor, torch.Tensor).
    """
    # Puntos interiores en [-3,3]x[-3,3]
    x_int = np.random.normal(loc=0.0, scale=2.5, size=(N_interior, 1))
    y_int = np.random.normal(loc=0.0, scale=2.5, size=(N_interior, 1))
    #y_int = np.clip(y_int, -100000, -0.01)  # fuerza a estar bajo el eje x

    X_int = np.hstack((x_int, y_int))  # (N_interior,2)
    X_int = torch.tensor(X_int, dtype=torch.float32, requires_grad=True,device=device)

    # Puntos sobre la lÃ­nea y = 0 (condiciÃ³n de frontera)
    
    x_bnd = np.random.normal(loc=0.0, scale=std_dev, size=(N_boundary, 1))
    y_bnd = np.zeros((N_boundary, 1))

    X_bnd = np.hstack((x_bnd, y_bnd))
    X_bnd = torch.tensor(X_bnd, dtype=torch.float32, requires_grad=True,device=device)
    
    return X_int, X_bnd
