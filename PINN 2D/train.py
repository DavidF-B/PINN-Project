import matplotlib.pyplot as plt
import torch
import torch.nn as nn



eps = 0.05
lr = 1e-3
epochs = 10000
N_r = 50
N_bc = 50

def q_eps(y):
    return (1-torch.exp((y-1)/eps))/(1-torch.exp(torch.tensor(-2/eps)))


class PINN(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2,20)
        self.linear2 = nn.Linear(20,20)
        self.linear3 = nn.Linear(20,1)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


# Boundary Loss
coord_add = torch.linspace(-1,1,N_bc).unsqueeze(1)
vec_bc_neg = torch.full_like(coord_add,-1)
vec_bc_pos = torch.full_like(coord_add,1)

# Clockwise numbering of the rectangle's sides, starting from the top
x_bc_1 = torch.cat((coord_add,vec_bc_pos),dim = 1)
x_bc_2 = torch.cat((vec_bc_pos, coord_add), dim = 1)
x_bc_3 = torch.cat((coord_add,vec_bc_neg),dim = 1)
x_bc_4 = torch.cat((vec_bc_neg,coord_add),dim = 1)

u_bc_1 = torch.zeros_like(coord_add)
u_bc_2 = q_eps(x_bc_2[:,1:2])
u_bc_3 = coord_add
u_bc_4 = -q_eps(x_bc_4[:,1:2])



# Points inside the domain

x_r_grid = torch.meshgrid(torch.linspace(-1+0.0001,1-0.0001,N_r),
                          torch.linspace(-1+0.0001,1-0.0001,N_r), indexing="ij")
x_r = torch.stack(x_r_grid,dim=-1) # shape (50,50,2)
x_r = x_r.reshape(-1,2)
x_r.requires_grad_(True)

# Instanciation du modèle

model = PINN()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

loss_tot_history = []
loss_bc_history = []
loss_domain_history = []

for epoch in range(epochs):

    optimizer.zero_grad()

    # Forward Pass
    u_bc_pred_1 = model(x_bc_1)
    u_bc_pred_2 = model(x_bc_2)
    u_bc_pred_3 = model(x_bc_3)
    u_bc_pred_4 = model(x_bc_4)
    u_r_pred = model(x_r)

    # Boundary Loss computation

    loss_bc = torch.mean((u_bc_pred_1-u_bc_1)**2) + torch.mean((u_bc_pred_2-u_bc_2)**2) + torch.mean((u_bc_pred_3-u_bc_3)**2) + torch.mean((u_bc_pred_4-u_bc_4)**2)

    # Domain Loss computation

    grad_u = torch.autograd.grad(inputs=x_r,outputs=u_r_pred,
                                 grad_outputs=torch.ones_like(u_r_pred), 
                                 create_graph=True)[0]

    du_dx, du_dy = grad_u[:,0:1], grad_u[:,1:2]

    d2u_dx2 = torch.autograd.grad(inputs = x_r, 
                                  outputs = du_dx,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True)[0][:,0:1]
    
    d2u_dy2 = torch.autograd.grad(inputs = x_r, 
                                  outputs = du_dy,
                                  grad_outputs=torch.ones_like(du_dy),
                                  create_graph=True)[0][:,1:2]

    delta_u = d2u_dx2 + d2u_dy2

    residual = -eps*delta_u + du_dy
    loss_domain = torch.mean(residual**2)

    loss_tot = loss_domain + loss_bc

    loss_bc_history.append(loss_bc.item())
    loss_domain_history.append(loss_domain.item())
    loss_tot_history.append(loss_tot.item())

    loss_tot.backward()

    optimizer.step()

    if (epoch+1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss Totale: {loss_tot.item():.6f} | Loss BC: {loss_bc.item():.6f} | Loss R: {loss_domain.item():.6f}")


torch.save(model.state_dict(), f"pinn_2D_eps_{eps}.pt")
print(f"Modèle sauvegardé sous 'pinn_2D_eps_{eps}.pt'")

# --- TRACÉ DE LA LOSS ---
plt.figure(figsize=(8, 5))
plt.plot(loss_tot_history, label="Loss Totale (L)", color="black", linewidth=2)
plt.plot(loss_bc_history, label="Loss Bords (L_u)", color="blue", linestyle="--")
plt.plot(loss_domain_history, label="Loss Domaine (L_r)", color="red", linestyle="--")

plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss (Échelle log)")
plt.title("Évolution des pertes pendant l'entraînement du PINN")
plt.legend()
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.savefig(f"Evolution_de_la_loss_2D_eps={eps}.png")
plt.show()





