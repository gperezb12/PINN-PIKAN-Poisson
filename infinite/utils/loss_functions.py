import torch
from torch.autograd import grad
from typing import Callable, Iterable


def loss_pde_inverse(model_u: Callable[[torch.Tensor], torch.Tensor], 
                     model_k: Callable[[torch.Tensor], torch.Tensor], 
                     X_int: torch.Tensor) -> torch.Tensor:
    """Compute PDE residual loss for inverse problem.
    
    Args:
        model_u: Neural network model for u(x,y)
        model_k: Neural network model for k(x,y) 
        X_int: Interior collocation points
        
    Returns:
        Mean squared PDE residual loss
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
    alpha = 0.25
    beta = 10
    epsilon = 1
    forcing = (epsilon*(exp(y/epsilon) + 1)*(3*exp(y/epsilon) + 1)*(4*alpha**2*x**2*cos(beta*y) + 4*alpha**2*y**2*cos(beta*y) + 4*alpha*beta*y*sin(beta*y) - 4*alpha*cos(beta*y) - beta**2*cos(beta*y))*exp(alpha*(x**2 + y**2)) - 2*(2*alpha*y*cos(beta*y) + beta*sin(beta*y))*exp((alpha*epsilon*(x**2 + y**2) + y)/epsilon))*exp(-2*alpha*(x**2 + y**2))/(epsilon*(exp(y/epsilon) + 1)**2)

    residual = k*(u_xx + u_yy) + (k_x * u_x + k_y * u_y) - forcing
    return torch.mean(residual**2)

def loss_bc(model: Callable[[torch.Tensor], torch.Tensor], 
            X_bnd: torch.Tensor, 
            alpha: float = 0.25, 
            beta: float = 7.5) -> torch.Tensor:
    """Compute boundary condition loss.
    
    Args:
        model: Neural network model
        X_bnd: Boundary points where y=0
        alpha: Parameter alpha in boundary condition
        beta: Parameter beta in boundary condition
        
    Returns:
        Mean squared error between prediction and boundary condition
    """ 
    x = X_bnd[:, 0:1]  # Solo la coordenada x, ya que y = 0
    u_true = 5.0*torch.exp(-alpha * x**2) * torch.cos(beta * x)
    u_pred = model(X_bnd)

    return torch.mean((u_pred - u_true) ** 2)

def loss_data_u(model: Callable[[torch.Tensor], torch.Tensor], 
               X_data: torch.Tensor, 
               u_data: torch.Tensor) -> torch.Tensor:
    """Compute data loss for u predictions.
    
    Args:
        model: Neural network model for u
        X_data: Input data points
        u_data: Target u values
        
    Returns:
        Mean squared error between predictions and data
    """
    u_pred = model(X_data)
    return torch.mean((u_pred - u_data)**2)

def loss_data_k(model: Callable[[torch.Tensor], torch.Tensor], 
               X_data: torch.Tensor, 
               k_data: torch.Tensor) -> torch.Tensor:
    """Compute data loss for k predictions.
    
    Args:
        model: Neural network model for k
        X_data: Input data points  
        k_data: Target k values
        
    Returns:
        Mean squared error between predictions and data
    """
    k_pred = model(X_data)
    return torch.mean((k_pred - k_data)**2)

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for _ in range(order):
        dy = torch.autograd.grad(
            dy, x,
            grad_outputs=torch.ones_like(dy),
            create_graph=True,
            retain_graph=True
        )[0]
    return dy

def regularization_loss(models: Iterable[torch.nn.Module], 
                        lambda_l1: float = 0.0, 
                        lambda_l2: float = 0.0) -> torch.Tensor:
    """Compute combined L1 and L2 regularization loss.
    
    Args:
        models: Iterable of models to regularize
        lambda_l1: L1 regularization weight
        lambda_l2: L2 regularization weight
        
    Returns:
        Scalar regularization loss tensor
    """
    models = list(models)
    device = None
    dtype = None
    for model in models:
        for param in model.parameters():
            device = param.device
            dtype = param.dtype
            break
        if device is not None:
            break
    if device is None:
        return torch.tensor(0.0)

    reg = torch.zeros((), device=device, dtype=dtype)
    if lambda_l1 == 0.0 and lambda_l2 == 0.0:
        return reg

    l1_term = torch.zeros_like(reg)
    l2_term = torch.zeros_like(reg)
    for model in models:
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if lambda_l1 != 0.0:
                l1_term = l1_term + param.abs().sum()
            if lambda_l2 != 0.0:
                l2_term = l2_term + param.pow(2).sum()

    return lambda_l1 * l1_term + lambda_l2 * l2_term
