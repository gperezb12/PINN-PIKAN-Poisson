import torch
from torch.autograd import grad


def loss_pde_inverse(model_u: torch.nn.Module, model_k: torch.nn.Module, X_int: torch.Tensor) -> torch.Tensor:
    """Compute PDE residual loss for inverse problem with variable diffusion coefficient.
    
    Args:
        model_u: Neural network predicting solution u(x,y)
        model_k: Neural network predicting diffusion coefficient k(x,y)
        X_int: Interior collocation points (N, 2)
    
    Returns:
        Mean squared PDE residual
    """
    u = model_u(X_int)
    k = model_k(X_int)

    grads_u = derivative(u, X_int, order=1)
    u_x = grads_u[:, 0:1]
    u_y = grads_u[:, 1:2]

    u_xx = derivative(u_x, X_int, order=1)[:, 0:1]
    u_yy = derivative(u_y, X_int, order=1)[:, 1:2]

    grads_k = derivative(k, X_int, order=1)
    k_x = grads_k[:, 0:1]
    k_y = grads_k[:, 1:2]

    x = X_int[:, 0:1]
    y = X_int[:, 1:2]
    
    pi = torch.pi
    sin = torch.sin
    exp = torch.exp
    cos = torch.cos
    sqrt = torch.sqrt
    alpha = 0.5
    beta = 10
    epsilon = 0.75
    forcing = (-4*alpha*y*exp((alpha*epsilon*(x**2 + y**2) - y - 1.5)/epsilon)*cos(beta*x) + epsilon*(exp((-y - 1.5)/epsilon) + 1)*(exp((-y - 1.5)/epsilon) + 3)*(4*alpha**2*x**2*cos(beta*x) + 4*alpha**2*y**2*cos(beta*x) + 4*alpha*beta*x*sin(beta*x) - 4*alpha*cos(beta*x) - beta**2*cos(beta*x))*exp(alpha*(x**2 + y**2)))*exp(-2*alpha*(x**2 + y**2))/(epsilon*(exp((-y - 1.5)/epsilon) + 1)**2)

    residual = k*(u_xx + u_yy) + (k_x * u_x + k_y * u_y) - forcing
    return torch.mean(residual**2)

def loss_bc(model: torch.nn.Module, X_bnd: torch.Tensor, alpha: float = 0.5, beta: float = 10) -> torch.Tensor:
    """Compute boundary condition loss at y=0.
    
    Args:
        model: Neural network predicting u(x,y)
        X_bnd: Boundary points at y=0 (N, 2)
        alpha: Parameter in analytic solution
        beta: Parameter in analytic solution
    
    Returns:
        Mean squared error between prediction and boundary condition
    """ 
    x = X_bnd[:, 0:1]  # Solo la coordenada x, ya que y = 0
    u_true = torch.exp(-alpha * x**2) * torch.cos(beta * x)
    u_pred = model(X_bnd)

    return torch.mean((u_pred - u_true) ** 2)

def loss_data_u(model: torch.nn.Module, X_data: torch.Tensor, u_data: torch.Tensor) -> torch.Tensor:
    """Compute data fitting loss for solution u.
    
    Args:
        model: Neural network predicting u(x,y)
        X_data: Data point coordinates (N, 2)
        u_data: Observed solution values (N, 1)
    
    Returns:
        Mean squared error between prediction and data
    """
    u_pred = model(X_data)
    return torch.mean((u_pred - u_data)**2)

def loss_data_k(model: torch.nn.Module, X_data: torch.Tensor, k_data: torch.Tensor) -> torch.Tensor:
    """Compute data fitting loss for diffusion coefficient k.
    
    Args:
        model: Neural network predicting k(x,y)
        X_data: Data point coordinates (N, 2)
        k_data: Observed diffusion coefficient values (N, 1)
    
    Returns:
        Mean squared error between prediction and data
    """
    k_pred = model(X_data)
    return torch.mean((k_pred - k_data)**2)

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute derivative using automatic differentiation.
    
    Args:
        dy: Function output tensor
        x: Input tensor with respect to which derivative is computed
        order: Order of derivative (1 for first derivative, 2 for second, etc.)
    
    Returns:
        Derivative tensor of specified order
    """
    for _ in range(order):
        dy = torch.autograd.grad(
            dy, x,
            grad_outputs=torch.ones_like(dy),
            create_graph=True,
            retain_graph=True
        )[0]
    return dy