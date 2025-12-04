import torch
from torch.autograd import grad


def loss_pde_inverse(model_u, model_k, X_int):
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

def loss_bc(model, X_bnd, alpha=0.5, beta=10):
    """
    Calcula el MSE entre la predicciÃ³n del modelo y la condiciÃ³n de frontera
    u(x,0) = exp(-alpha * x^2) * cos(beta * x).

    Args:
        model: Red neuronal.
        X_bnd (torch.Tensor): Puntos sobre la frontera y=0.
        alpha (float): ParÃ¡metro alpha.
        beta (float): ParÃ¡metro beta.

    Returns:
        MSE (torch.Tensor)
    """ 
    x = X_bnd[:, 0:1]  # Solo la coordenada x, ya que y = 0
    u_true = torch.exp(-alpha * x**2) * torch.cos(beta * x)
    u_pred = model(X_bnd)

    return torch.mean((u_pred - u_true) ** 2)

def loss_data_u(model, X_data, u_data):
    u_pred = model(X_data)
    return torch.mean((u_pred - u_data)**2)

def loss_data_k(model, X_data, k_data):
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