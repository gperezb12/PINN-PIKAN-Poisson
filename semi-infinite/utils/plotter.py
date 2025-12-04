
def plot_solution_and_k(modelU, modelK, epoch, folder="figs_inverse_mixed", n_points=150):
    if not os.path.exists(folder):
        os.makedirs(folder)

    x_vals = np.linspace(-9, 9, n_points)
    y_vals = np.linspace(-9, 0, n_points)
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
    epsilon = 0.5

    u_true = np.exp(-alpha * (X_mesh**2 + Y_mesh**2)) * np.cos(beta * X_mesh)
    k_true =  1 + 2 / (1 + np.exp(-(Y_mesh + 1.5) / epsilon))

    # Create a 2D figure with 4 subplots
    fig = plt.figure(figsize=(12, 10))
    
    # 2D plot for PINN u
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.pcolormesh(X_mesh, Y_mesh, u_pred, cmap='viridis', shading='auto')
    ax1.set_title(f"PINN u (epoch {epoch})")
    fig.colorbar(im1, ax=ax1)
    
    # 2D plot for Analytic u
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.pcolormesh(X_mesh, Y_mesh, u_true, cmap='viridis', shading='auto')
    ax2.set_title("Analytic u")
    fig.colorbar(im2, ax=ax2)
    
    # 2D plot for PINN k
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.pcolormesh(
        X_mesh, Y_mesh, k_pred,
        cmap='viridis', shading='auto',
        vmin=2.5, vmax=5.5
    )
    ax3.set_title(f"PINN k (epoch {epoch})")
    cbar = fig.colorbar(im3, ax=ax3)
    
    # 2D plot for Analytic k
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.pcolormesh(X_mesh, Y_mesh, k_true, cmap='viridis', shading='auto',vmin=2.5, vmax=5.5)
    ax4.set_title("Analytic k")
    fig.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"solution_epoch_{epoch}.png"))
    plt.close(fig)
