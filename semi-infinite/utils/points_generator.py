
import numpy as np
import torch




def generate_interior_data(N_data: int = 100, device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate interior data points with analytic solution values.
    
    Args:
        N_data: Number of data points to generate
        device: Device to place tensors on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (X_data, u_data, k_data) tensors
    """
    x_data = np.random.normal(loc=0.0, scale=2.5, size=(N_data, 1))
    y_data = -np.random.exponential(scale=1.25, size=(N_data, 1))
    X_data_np = np.hstack((x_data, y_data))
    
    # Solución analítica
    alpha = 0.5
    beta = 10
    epsilon = 0.75
    
    u_data_np = np.exp(-alpha * (x_data**2 + y_data**2)) * np.cos(beta * x_data)
    k_true =  1 + 2 / (1 + np.exp(-(y_data + 1.5) / epsilon))
    # Agregar ruido del 5%
    u_noise = 0.2 * np.random.randn(*u_data_np.shape) * np.abs(u_data_np)
    k_noise = 0.2 * np.random.randn(*k_true.shape) * np.abs(k_true)
    
    u_data_np_noisy = u_data_np 
    k_data_np_noisy = k_true 

    # Convertir a tensores
    X_data = torch.tensor(X_data_np, dtype=torch.float32, requires_grad=True, device=device)
    k_data = torch.tensor(k_data_np_noisy, dtype=torch.float32, requires_grad=True, device=device)
    u_data = torch.tensor(u_data_np_noisy, dtype=torch.float32, device=device)
    
    return X_data, u_data, k_data

def generate_collocation_points(N_interior: int = 2000, N_boundary: int = 200, std_dev: float = 2.5, device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    """Generate collocation points for PDE residual and boundary conditions.
    
    Args:
        N_interior: Number of interior collocation points
        N_boundary: Number of boundary points at y=0
        std_dev: Standard deviation for normal distribution sampling
        device: Device to place tensors on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (X_int, X_bnd) tensors for interior and boundary points
    """
    # Puntos interiores en [-3,3]x[-3,3]
    x_int = np.random.normal(loc=0.0, scale=2.5, size=(N_interior, 1))
    y_int = -np.random.exponential(scale=1.25, size=(N_interior, 1))
    #y_int = np.random.normal(loc=0.0, scale=2.5, size=(N_interior, 1))

    #y_int = np.clip(y_int, -100000, -0.01)  # fuerza a estar bajo el eje x

    X_int = np.hstack((x_int, y_int))  # (N_interior,2)
    X_int = torch.tensor(X_int, dtype=torch.float32, requires_grad=True, device=device)

    # Puntos sobre la lÃ­nea y = 0 (condiciÃ³n de frontera)
    
    x_bnd = np.random.normal(loc=0.0, scale=std_dev, size=(N_boundary, 1))
    y_bnd = np.zeros((N_boundary, 1))

    X_bnd = np.hstack((x_bnd, y_bnd))
    X_bnd = torch.tensor(X_bnd, dtype=torch.float32, requires_grad=True, device=device)
    
    return X_int, X_bnd
