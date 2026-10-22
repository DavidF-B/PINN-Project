import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path



eps = 0.001
shift = -1.0
scale = 1000.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):

    def __init__(self, shift=shift):
        super().__init__()
        self.shift = shift
        self.scale = scale
        self.linear1 = nn.Linear(1,20)
        self.linear2 = nn.Linear(20,20)
        self.linear3 = nn.Linear(20,1)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.scale*(x + self.shift)
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

model = PINN().to(device)

checkpoint = Path(__file__).resolve().parent / f"pinn_eps_{eps}_shift_{shift}_scale_{scale}.pt"
model.load_state_dict(torch.load(checkpoint, weights_only=True))
model.eval()  

print("Poids du modèle chargés avec succès !")


# --- TRACÉ DE LA SOLUTION APPRISE ---

x_test_np = np.linspace(0.0,1.0,1000)
x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32, device=device).unsqueeze(1)


with torch.no_grad():
     u_pred_tensor = model(x_test_tensor)

u_pred_np = u_pred_tensor.cpu().numpy().flatten()

u_exacte_np = (np.exp((x_test_np - 1.0) / eps) - np.exp(-1.0 / eps)) / (1.0 - np.exp(-1.0 / eps))

#   Note sur la stabilité numérique :
#    La forme brute (1 - exp(x/eps)) / (1 - exp(1/eps)) provoque un overflow
#    pour eps <= 1e-3 (ex: exp(1000) -> Inf -> NaN).
#    
#    En factorisant par exp(-1/eps), on transforme les termes en exponentielles
#    à arguments négatifs ou nuls ((x-1)/eps <= 0), garantissant une évaluation
#    stable sans overflow, où exp(-1/eps) sous-flow proprement vers 0.


# --- TRACÉ DE LA VUE GLOBALE ---
plt.figure(figsize=(8, 5))

plt.plot(x_test_np, u_exacte_np, 'k-', linewidth=2, label="Solution Analytique Exacte")

plt.plot(x_test_np, u_pred_np, 'r--', linewidth=2, label="Prédiction PINN")

plt.xlabel("x")
plt.ylabel("T(x)")
plt.title(f"Comparaison PINN vs Solution Analytique (eps = {eps}, shift={shift}_scale={scale})")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig(f"Comparaison_PINN_vs_Solution Analytique_eps_{eps}__shift_{shift}_scale={scale}.png")
plt.show()


     

     