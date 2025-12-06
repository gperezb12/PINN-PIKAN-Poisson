import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
from typing import Callable, Tuple

def plot_solution_and_k(modelU: Callable[[torch.Tensor], torch.Tensor], 
                       modelK: Callable[[torch.Tensor], torch.Tensor], 
                       epoch: int, 
                       folder: str = "figs_inverse_mixed", 
                       n_points: int = 250, 
                       device: str = 'cuda') -> None:
    """Plot solution u and parameter k predictions.
    
    Args:
        modelU: Neural network model for u
        modelK: Neural network model for k
        epoch: Current training epoch
        folder: Output folder for plots
        n_points: Number of grid points
        device: Device to run computations on
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    x_vals = np.linspace(-3, 3, n_points)
    y_vals = np.linspace(-3, 3, n_points)
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    XY_np = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T
    XY_torch = torch.tensor(XY_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = modelU(XY_torch).cpu().numpy()
        k_pred = modelK(XY_torch).cpu().numpy()
    u_pred = u_pred.reshape(n_points, n_points)
    k_pred = k_pred.reshape(n_points, n_points)
    alpha = 0.5
    beta = 10
    epsilon = 1

    u_true = np.exp(-alpha * (X_mesh**2 + Y_mesh**2)) * np.cos(beta * Y_mesh)
    k_true = 1 + 2 / (1 + np.exp(-Y_mesh / epsilon))

    figU, axU = plot_styled_waves(X_mesh, Y_mesh, u_pred, title=f"u aproximation PINN  (Epoch {epoch})", saveas=f"figsKan/figs(6,12)/epoch_{epoch}/solution_u.pdf")
    # 2D plot for PINN u
    figK, axK = plot_styled_k(X_mesh, Y_mesh, k_pred, title=f"k aproximation PINN  (Epoch {epoch})", saveas=f"figsKan/figs(6,12)/epoch_{epoch}/solution_k.pdf")
    
    """
    # 2D plot for PINN k
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.pcolormesh(
        X_mesh, Y_mesh, k_pred,
        cmap='GnBu', shading='auto'
    )
    ax3.set_title(f"PINN k (epoch {epoch})")
    cbar = fig.colorbar(im3, ax=ax3)
    
    # 2D plot for Analytic k
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.pcolormesh(X_mesh, Y_mesh, k_true, cmap='GnBu', shading='auto')
    ax4.set_title("Analytic k")
    fig.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"solution_epoch_{epoch}.png"))
    plt.close(fig)
    """

def plot_styled_waves(X: np.ndarray, 
                     Y: np.ndarray, 
                     Z: np.ndarray, 
                     title: str = "Aqui va el titulo", 
                     saveas: str = "figure.svg", 
                     epoch: int = 0) -> Tuple[plt.Figure, plt.Axes]:
    """Create a styled wave plot.
    
    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        Z: Values to plot
        title: Plot title
        saveas: Save path
        epoch: Training epoch
        
    Returns:
        Tuple of (figure, axes)
    """
    # Set the font to Computer Modern
    """
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "font.size": 14
    })
    """
    fig, ax = plt.subplots(figsize=(5, 5.75))
    cmap = plt.get_cmap('RdBu')
    
    vmax = np.abs(Z).max()
    vmax = 1
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', vmin=-vmax, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.07, shrink=1)
    cbar.set_ticks([-vmax, 0, vmax])
    im.set_clim(-vmax, vmax)
    
    # Axis formatting
    ax.set_xticks([0])
    ax.set_xticklabels([''])
    ax.set_yticks([0])
    ax.set_yticklabels([''])
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    plt.title(title, pad=15, fontsize=22)
    os.makedirs(os.path.dirname(saveas), exist_ok=True)

    plt.savefig(saveas)
    plt.close(fig)

    return fig, ax

def plot_styled_k(X: np.ndarray, 
                 Y: np.ndarray, 
                 Z: np.ndarray, 
                 title: str = "Aqui va el titulo", 
                 saveas: str = "figure.svg", 
                 epoch: int = 0) -> Tuple[plt.Figure, plt.Axes]:
    """Create a styled k parameter plot.
    
    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        Z: Values to plot
        title: Plot title
        saveas: Save path
        epoch: Training epoch
        
    Returns:
        Tuple of (figure, axes)
    """
    # Set the font to Computer Modern

    
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "font.size": 14
    })
    
    fig, ax = plt.subplots(figsize=(5, 5.75))
    cmap = plt.get_cmap('GnBu')
    
    vmax = Z.max()
    vmin = Z.min()
    vmax = 3
    vmin = 1

    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', vmin=-vmax, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.07, shrink=1)
    cbar.set_ticks([vmin, 0, vmax])
    im.set_clim(vmin, vmax)
    
    # Axis formatting
    ax.set_xticks([0])
    ax.set_xticklabels([''])
    ax.set_yticks([0])
    ax.set_yticklabels([''])
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    plt.title(title, pad=15, fontsize=22)
    os.makedirs(os.path.dirname(saveas), exist_ok=True)

    plt.savefig(saveas)
    plt.close(fig)
    return fig, ax




def plot_styled_error(X: np.ndarray, 
                     Y: np.ndarray, 
                     Z: np.ndarray, 
                     title: str = "Aqui va el titulo", 
                     saveas: str = "figure.svg", 
                     epoch: int = 0) -> Tuple[plt.Figure, plt.Axes]:
    """Create a styled error plot.
    
    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        Z: Error values to plot
        title: Plot title
        saveas: Save path
        epoch: Training epoch
        
    Returns:
        Tuple of (figure, axes)
    """
    # Set the font to Computer Modern
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "font.size": 14
    })
    
    fig, ax = plt.subplots(figsize=(5, 5.75))
    cmap = plt.get_cmap('PiYG')
    
    vmax = abs(Z).max()
    vmax = 0.1
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', vmin=-vmax, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.07, shrink=1)
    cbar.set_ticks([-vmax, 0, vmax])
    im.set_clim(-vmax, vmax)
    
    # Axis formatting
    ax.set_xticks([0])
    ax.set_xticklabels([''])
    ax.set_yticks([0])
    ax.set_yticklabels([''])
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    plt.title(title, pad=15, fontsize=22)
    plt.savefig(saveas)
    return fig, ax


def plot_linear(x: np.ndarray, 
               y: np.ndarray, 
               titleofplot: str, 
               yaxis: str = None, 
               xaxis: str = None, 
               epoch: int = 0) -> None:
    """Plot linear data with custom styling.
    
    Args:
        x: X-axis data
        y: Y-axis data
        titleofplot: Plot title
        yaxis: Y-axis label (optional)
        xaxis: X-axis label (optional)
        epoch: Training epoch
    """
    # Set seaborn style first
    sns.set_theme(style="darkgrid")
    
    # Then update rcParams (this will override Seaborn's changes)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "font.size": 14
    })
    
    # Plot
    plt.plot(x, y, 'b-')
    
    # Set title
    plt.title(titleofplot, pad=10, fontsize=16)
    
    # Set axis labels only if provided
    if xaxis is not None:
        plt.xlabel(xaxis)
    if yaxis is not None:
        plt.ylabel(yaxis)
    


def plot_log(x: np.ndarray, 
            y: np.ndarray, 
            titleofplot: str, 
            yaxis: str = None, 
            xaxis: str = None, 
            epoch: int = 0) -> None:
    """Plot data with log scale and custom styling.
    
    Args:
        x: X-axis data
        y: Y-axis data
        titleofplot: Plot title
        yaxis: Y-axis label (optional)
        xaxis: X-axis label (optional)
        epoch: Training epoch
    """
    # Set seaborn style first
    sns.set_theme(style="darkgrid")
    
    # Then update rcParams (this will override Seaborn's changes)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "font.size": 14
    })
    
    # Plot
    plt.plot(x, y, 'b-')
    plt.yscale('log')
    plt.xscale('log')
    # Set title
    plt.title(titleofplot, pad=10, fontsize=16)
    
    # Set axis labels only if provided
    if xaxis is not None:
        plt.xlabel(xaxis)
    if yaxis is not None:
        plt.ylabel(yaxis)
     

def plot_styled_waves_sq(X: np.ndarray, 
                        Y: np.ndarray, 
                        Z: np.ndarray, 
                        title: str = "Aqui va el titulo", 
                        saveas: str = "figure.svg", 
                        epoch: int = 0) -> Tuple[plt.Figure, plt.Axes]:
    """Create a styled wave plot with square boundary.
    
    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        Z: Values to plot
        title: Plot title
        saveas: Save path
        epoch: Training epoch
        
    Returns:
        Tuple of (figure, axes)
    """
    # Set the font to Computer Modern
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "font.size": 14
    })
    
    fig, ax = plt.subplots(figsize=(5, 5.75))
    cmap = plt.get_cmap('RdBu')
    
    vmax = np.abs(Z).max()
    vmax = 1
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', vmin=-vmax, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.07, shrink=1)
    cbar.set_ticks([-vmax, 0, vmax])
    im.set_clim(-vmax, vmax)
    
    # Axis formatting
    ax.set_xticks([0])
    ax.set_xticklabels([''])
    ax.set_yticks([0])
    ax.set_yticklabels([''])
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    # Add purple square
    square_x = [-3, 3, 3, -3, -3]
    square_y = [-3, -3, 0, 0, -3]

    ax.plot(square_x, square_y, color='purple', linewidth=2)

    plt.title(title, pad=10, fontsize=16)
    os.makedirs(os.path.dirname(saveas), exist_ok=True)

    plt.savefig(saveas)
    plt.close(fig)

    return fig, ax


def plot_styled_k_sq(X: np.ndarray, 
                    Y: np.ndarray, 
                    Z: np.ndarray, 
                    title: str = "Aqui va el titulo", 
                    saveas: str = "figure.svg", 
                    epoch: int = 0) -> Tuple[plt.Figure, plt.Axes]:
    """Create a styled k parameter plot with square boundary.
    
    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        Z: Values to plot
        title: Plot title
        saveas: Save path
        epoch: Training epoch
        
    Returns:
        Tuple of (figure, axes)
    """
    # Set the font to Computer Modern
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "font.size": 14
    })
    
    fig, ax = plt.subplots(figsize=(5, 5.75))
    cmap = plt.get_cmap('GnBu')
    
    vmax = Z.max()
    vmin = Z.min()
    vmax = 3
    vmin = 1

    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.07, shrink=1)
    cbar.set_ticks([vmin, vmax])
    im.set_clim(vmin, vmax)
    
    # Axis formatting
    ax.set_xticks([0])
    ax.set_xticklabels([''])
    ax.set_yticks([0])
    ax.set_yticklabels([''])
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    # Add purple square
    square_x = [-3, 3, 3, -3, -3]
    square_y = [-3, -3, 3, 3, -3]
    ax.plot(square_x, square_y, color='purple', linewidth=2)

    plt.title(title, pad=10, fontsize=16)
    os.makedirs(os.path.dirname(saveas), exist_ok=True)

    plt.savefig(saveas)
    plt.close(fig)
    return fig, ax
