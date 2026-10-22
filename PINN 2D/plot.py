import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

eps = 0.05

def q_eps(y):
    return (1 - np.exp((y - 1) / eps)) / (1 - np.exp(-2 / eps))

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        return self.linear3(x)

model = PINN()
model.load_state_dict(torch.load(f"pinn_2D_eps_{eps}.pt", weights_only=True))
model.eval()


N = 200
xs = np.linspace(-1, 1, N)
X, Y = np.meshgrid(xs, xs, indexing="ij")            
pts = np.stack((X, Y), axis=-1).reshape(-1, 2)       

with torch.no_grad():
    u_pred = model(torch.tensor(pts, dtype=torch.float32)).numpy().reshape(N, N)

u_exact = X * q_eps(Y)                            
err = np.abs(u_pred - u_exact)


fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

for ax, Z, title in zip(
    axs, [u_exact, u_pred, err], ["Solution exacte", "PINN", "|erreur|"]
):
    c = ax.contourf(X, Y, Z, 50, cmap="viridis")

    
    cbar = fig.colorbar(c, ax=ax, pad=0.03)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")  
plt.suptitle(
    f"PINN vs solution analytique (eps = {eps})", fontsize=14, fontweight="bold"
)
plt.savefig(
    f"Comparaison_PINN_vs_Solution_Analytique_2D_eps={eps}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Surface 3D
surf = ax.plot_surface(X, Y, u_pred, cmap="viridis", edgecolor="none", alpha=0.9)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("T(x,y)")
ax.set_title(f"Surface 3D du PINN (eps = {eps})")


ax.view_init(elev=25, azim=-45)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.savefig(f"Surface_3D_eps_{eps}.png", bbox_inches="tight", dpi=300)
plt.show()



y_cut = Y[
    -1, :
]  
u_exact_cut = u_exact[-1, :]
u_pred_cut = u_pred[-1, :]

plt.figure(figsize=(7, 4.5))
plt.plot(
    y_cut,
    u_exact_cut,
    "r-",
    label="Solution exacte $T(1,y) = q_\\varepsilon(y)$",
    linewidth=2,
)
plt.plot(
    y_cut,
    u_pred_cut,
    "b--",
    label="Prédiction PINN",
    linewidth=1.5,
)

# Zone de la couche limite
plt.axvspan(
    1 - eps,
    1,
    color="red",
    alpha=0.15,
    label=f"Couche limite (épaisseur $\\delta \\approx \\varepsilon = {eps}$)",
)

plt.xlabel("y")
plt.ylabel("T(1, y)")
plt.title(f"Coupe 1D à x = 1 (eps = {eps})")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig(f"Coupe_1D_y_eps_{eps}.png", bbox_inches="tight", dpi=300)
plt.show()