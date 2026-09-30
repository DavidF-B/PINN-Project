import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


eps = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1,20)
        self.linear2 = nn.Linear(20,20)
        self.linear3 = nn.Linear(20,1)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

model = PINN().to(device)

model.load_state_dict(torch.load(f"premier_pinn_eps={eps}.pt", weights_only=True))
model.eval()  

print("Poids du modèle chargés avec succès !")


# --- TRACÉ DE LA SOLUTION APPRISE ---

x_test_np = np.linspace(0.0,1.0,1000)
x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32, device=device).unsqueeze(1)


with torch.no_grad():
     u_pred_tensor = model(x_test_tensor)

u_pred_np = u_pred_tensor.cpu().numpy().flatten()

u_exacte_np = (np.exp(x_test_np/eps)-1)/(np.exp(1/eps)-1)


# --- TRACÉ DE LA VUE GLOBALE ---
plt.figure(figsize=(8, 5))

plt.plot(x_test_np, u_exacte_np, 'k-', linewidth=2, label="Solution Analytique Exacte")

plt.plot(x_test_np, u_pred_np, 'r--', linewidth=2, label="Prédiction PINN")

plt.xlabel("x")
plt.ylabel("T(x)")
plt.title(f"Comparaison PINN vs Solution Analytique (eps = {eps})")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig(f"Comparaison PINN vs Solution Analytique_eps={eps}.png")
plt.show()


     

     