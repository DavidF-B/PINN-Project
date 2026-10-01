import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn



# --- HYPER-PARAMETRES ---
eps = 0.01
N_r = 500
lr = 1e-3
epochs = 10000
shift = -1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du matériel : {device}")


class PINN(nn.Module):

    def __init__(self, shift = shift):
        super().__init__()
        self.shift = shift
        self.linear1 = nn.Linear(1,20)
        self.linear2 = nn.Linear(20,20)
        self.linear3 = nn.Linear(20,1)
        self.tanh = nn.Tanh()

    def forward(self,x):

        x = x + self.shift
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

# --- DONNEES D'ENTREE ---
# Points au bord
x_bc = torch.tensor([[0.0],[1.0]], device = device)
u_bc = torch.tensor([[0.0],[1.0]], device = device)

# Points à l'intérieur du domaine
x_r = torch.linspace(0.001, 0.999, N_r, device = device, requires_grad=True).unsqueeze(1)

loss_history = []
loss_bc_history = []
loss_r_history = []



# --- BOUCLE D'ENTRAINEMENT ---
for epoch in range(epochs):

    optimizer.zero_grad()

    u_r = model(x_r)

    u_pred_bc = model(x_bc)

    loss_bc = torch.mean((u_pred_bc-u_bc)**2)

    du_dx = torch.autograd.grad(
        outputs = u_r, 
        inputs = x_r, 
        grad_outputs=torch.ones_like(u_r), 
        create_graph=True)[0]

    d2u_dx2 = torch.autograd.grad(
        outputs = du_dx, 
        inputs = x_r, 
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True)[0]

    residual = eps*d2u_dx2 - du_dx
    loss_r = torch.mean(residual**2)

    loss = loss_bc + loss_r

    loss_history.append(loss.item())
    loss_bc_history.append(loss_bc.item())
    loss_r_history.append(loss_r.item())
    

    loss.backward()

    optimizer.step()

    #scheduler.step()

    if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss Totale: {loss.item():.6f} | Loss BC: {loss_bc.item():.6f} | Loss R: {loss_r.item():.6f}")


# --- SAUVEGARDE DES POIDS DU MODÈLE ---
torch.save(model.state_dict(), f"pinn_eps={eps}_shift={shift}.pt")
print(f"Modèle sauvegardé sous 'premier_pinn_eps={eps}_shift={shift}.pt'")

# --- TRACÉ DE LA LOSS ---
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Loss Totale (L)", color="black", linewidth=2)
plt.plot(loss_bc_history, label="Loss Bords (L_u)", color="blue", linestyle="--")
plt.plot(loss_r_history, label="Loss Domaine (L_r)", color="red", linestyle="--")

plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss (Échelle log)")
plt.title("Évolution des pertes pendant l'entraînement du PINN")
plt.legend()
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.savefig(f"Evolution de la loss_eps={eps}_shift={shift}.png")
plt.show()


