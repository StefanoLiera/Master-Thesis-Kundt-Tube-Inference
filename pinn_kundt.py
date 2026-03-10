# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# %%
# Hyperparameters

# =========================
# Reproducibility
# =========================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# Hyperparameters
# =========================
N_HIDDEN_LAYERS = 10
N_NEURONS = 20
EPOCHS = 200
LEARNING_RATE = 0.005
BATCH_SIZE = 5

LR_DECAY_FACTOR = 0.5
LR_DECAY_STEP = 50

LAMBDA_DATA = 1.0
LAMBDA_PDE = 0.0


# =========================
# Physical parameters
# =========================
f = 400.0         # Hz
c0 = 343.0        # m/s
omega = 2.0 * np.pi * f
k = omega / c0

R_MIN, R_MAX = 0.0, 0.05
Z_MIN, Z_MAX = 0.0, 0.60

N_COLLOCATION = 1000
EPS_R = 1e-6


print("k =", k)




# %%
# Load dataset


# =========================
# Load dataset from COMSOL
# =========================
file_path = "v0.025_GP_Extrafine_3Colonne.txt"

print("Working directory:", os.getcwd())
print("File exists:", os.path.exists(file_path))

df = pd.read_csv(
    file_path,
    comment="%",
    header=None,
    sep=r"\s+",
    engine="python"
)

print("\nRaw dataframe:")
print(df.head())
print("Shape:", df.shape)


df.columns = ["Rdata", "Zdata", "Pdata"]

# Convert COMSOL notation from i to j
df["Pdata"] = df["Pdata"].astype(str).str.replace("i", "j", regex=False)

# Convert to Python complex numbers
df["Pdata"] = df["Pdata"].apply(complex)

# Extract real and imaginary parts
df["Preal"] = np.real(df["Pdata"])
df["Pimag"] = np.imag(df["Pdata"])



print(df.head())
print(df.dtypes)
print("Max abs(Preal):", np.max(np.abs(df["Preal"])))
print("Max abs(Pimag):", np.max(np.abs(df["Pimag"])))

print("\nRenamed dataframe:")
print(df.head())
print("Columns:", df.columns.tolist())



# %%
# Dataset split


# =========================
# Split dataset: 70% train, 20% val, 10% test
# =========================
X = df[["Rdata", "Zdata"]].values
# y = df[["Pdata"]].values


# First split: 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Second split: from the remaining 30%, split into 20% val and 10% test
# 20/30 = 2/3 for validation, 10/30 = 1/3 for test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42
)

print("\nDataset split:")
print("Train shapes:", X_train.shape, y_train.shape)
print("Val shapes  :", X_val.shape, y_val.shape)
print("Test shapes :", X_test.shape, y_test.shape)


# %%
# =========================
# Normalize inputs and output
# =========================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled   = x_scaler.transform(X_val)
X_test_scaled  = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled   = y_scaler.transform(y_val)
y_test_scaled  = y_scaler.transform(y_test)

print("\nInput scaler mean:", x_scaler.mean_)
print("Input scaler std :", x_scaler.scale_)
print("Output scaler mean:", y_scaler.mean_)
print("Output scaler std :", y_scaler.scale_)



# %%

# =========================
# Convert to PyTorch tensors
# =========================
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

print("\nTensor shapes:")
print("X_train_t:", X_train_t.shape)
print("y_train_t:", y_train_t.shape)
print("X_val_t  :", X_val_t.shape)
print("y_val_t  :", y_val_t.shape)
print("X_test_t :", X_test_t.shape)
print("y_test_t :", y_test_t.shape)

# =========================
# Create PyTorch datasets and dataloaders
# =========================
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\nNumber of batches:")
print("Train batches:", len(train_loader))
print("Val batches  :", len(val_loader))
print("Test batches :", len(test_loader))


# =========================
# Quick plot of the FEM dataset
# =========================
plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    df["Zdata"],
    df["Rdata"],
    c=df["Pdata"],
    cmap="viridis"
)
plt.colorbar(scatter, label="Acoustic Pressure p")
plt.xlabel("Z [m]")
plt.ylabel("R [m]")
plt.title("FEM dataset exported from COMSOL")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Histograms of the target before and after scaling
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(y_train, bins=30)
plt.title("Original training pressure")
plt.xlabel("Pdata")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(y_train_scaled, bins=30)
plt.title("Scaled training pressure")
plt.xlabel("Scaled Pdata")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# %%

# Collocation points for PDE residual
# =========================
R_col = np.random.uniform(R_MIN, R_MAX, N_COLLOCATION)
Z_col = np.random.uniform(Z_MIN, Z_MAX, N_COLLOCATION)

# Avoid exactly R = 0 because of the 1/R term in axisymmetric Helmholtz
R_col = np.clip(R_col, EPS_R, None)

X_col = np.column_stack([R_col, Z_col])
X_col_scaled = x_scaler.transform(X_col)

X_col_t = torch.tensor(
    X_col_scaled,
    dtype=torch.float32,
    requires_grad=True
).to(device)

print("Collocation tensor shape:", X_col_t.shape)


# %%
# =========================
# PINN architecture
# =========================
class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=10, neurons=20):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


model = PINN(
    input_dim=2,
    output_dim=1,   # only Pimag for now
    hidden_layers=N_HIDDEN_LAYERS,
    neurons=N_NEURONS
).to(device)

print(model)


# %%
# =========================
# Forward test
# =========================
sample_out = model(X_train_t[:5])
print("Sample output shape:", sample_out.shape)
print(sample_out)



# %%
# =========================
# Loss, optimizer, scheduler
# =========================
mse_loss = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=LR_DECAY_STEP,
    gamma=LR_DECAY_FACTOR
)

print("Optimizer:", optimizer)
print("Scheduler:", scheduler)




# %%

# PDE residual for axisymmetric Helmholtz
# =========================
def pde_residual_axisymmetric(model, x_scaled, x_scaler, y_scaler, k, eps_r=1e-6):
    """
    Axisymmetric Helmholtz residual:
    d2p/dR2 + (1/R) dp/dR + d2p/dZ2 + k^2 p = 0

    x_scaled: normalized coordinates [R, Z]
    model output: normalized pressure (here only Pimag)
    """

    x_scaled.requires_grad_(True)

    # Network output in scaled space
    p_scaled = model(x_scaled)

    # Convert output back to physical scale
    y_std = torch.tensor(y_scaler.scale_[0], dtype=torch.float32, device=device)
    y_mean = torch.tensor(y_scaler.mean_[0], dtype=torch.float32, device=device)
    p = p_scaled * y_std + y_mean

    # Recover physical coordinates from scaled inputs
    x_mean = torch.tensor(x_scaler.mean_, dtype=torch.float32, device=device)
    x_std = torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device)

    x_phys = x_scaled * x_std + x_mean
    R = x_phys[:, 0:1]
    Z = x_phys[:, 1:2]

    R = torch.clamp(R, min=eps_r)

    # First derivatives wrt scaled coordinates
    grads = torch.autograd.grad(
        outputs=p,
        inputs=x_scaled,
        grad_outputs=torch.ones_like(p),
        create_graph=True,
        retain_graph=True
    )[0]

    dp_dR_scaled = grads[:, 0:1]
    dp_dZ_scaled = grads[:, 1:2]

    # Convert to derivatives wrt physical coordinates
    dR_dscaled = x_std[0]
    dZ_dscaled = x_std[1]

    dp_dR = dp_dR_scaled / dR_dscaled
    dp_dZ = dp_dZ_scaled / dZ_dscaled

    # Second derivatives
    d2p_dR2_scaled = torch.autograd.grad(
        outputs=dp_dR_scaled,
        inputs=x_scaled,
        grad_outputs=torch.ones_like(dp_dR_scaled),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]

    d2p_dZ2_scaled = torch.autograd.grad(
        outputs=dp_dZ_scaled,
        inputs=x_scaled,
        grad_outputs=torch.ones_like(dp_dZ_scaled),
        create_graph=True,
        retain_graph=True
    )[0][:, 1:2]

    d2p_dR2 = d2p_dR2_scaled / (dR_dscaled ** 2)
    d2p_dZ2 = d2p_dZ2_scaled / (dZ_dscaled ** 2)

    residual = d2p_dR2 + (1.0 / R) * dp_dR + d2p_dZ2 + (k ** 2) * p

    return residual


# %%
# =========================
# PDE residual test
# =========================
residual_test = pde_residual_axisymmetric(
    model=model,
    x_scaled=X_col_t[:10].clone(),
    x_scaler=x_scaler,
    y_scaler=y_scaler,
    k=k,
    eps_r=EPS_R
)

print("Residual shape:", residual_test.shape)
print(residual_test[:5])


# %%
# =========================
# PINN Training Loop
# =========================

history = {
    "train_total": [],
    "train_data": [],
    "train_pde": [],
    "val_data": []
}

best_val_loss = np.inf
best_model_path = "best_pinn_model.pt"

for epoch in range(EPOCHS):

    model.train()

    running_total = 0.0
    running_data = 0.0
    running_pde = 0.0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        # ------------------------
        # Data loss (FEM points)
        # ------------------------
        pred = model(xb)
        loss_data = mse_loss(pred, yb)

        # ------------------------
        # PDE loss (collocation points)
        # ------------------------
        residual = pde_residual_axisymmetric(
            model,
            X_col_t.clone(),
            x_scaler,
            y_scaler,
            k,
            eps_r=EPS_R
        )

        loss_pde = torch.mean(residual**2)

        # ------------------------
        # Total loss
        # ------------------------
        loss = LAMBDA_DATA * loss_data + LAMBDA_PDE * loss_pde

        loss.backward()
        optimizer.step()

        running_total += loss.item()
        running_data += loss_data.item()
        running_pde += loss_pde.item()

    scheduler.step()

    # ------------------------
    # Validation
    # ------------------------
    model.eval()

    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = mse_loss(val_pred, y_val_t).item()

    n_batches = len(train_loader)

    epoch_total = running_total / n_batches
    epoch_data = running_data / n_batches
    epoch_pde = running_pde / n_batches

    history["train_total"].append(epoch_total)
    history["train_data"].append(epoch_data)
    history["train_pde"].append(epoch_pde)
    history["val_data"].append(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch:4d} | "
            f"Train Total: {epoch_total:.4e} | "
            f"Data: {epoch_data:.4e} | "
            f"PDE: {epoch_pde:.4e} | "
            f"Val: {val_loss:.4e}"
        )

print("\nBest validation loss:", best_val_loss)


# %%
# =========================
# Load best model
# =========================

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print("Best model loaded.")

# %%
# =========================
# Test evaluation
# =========================

with torch.no_grad():

    y_test_pred_scaled = model(X_test_t).cpu().numpy()

y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)
y_test_true = y_test

rmse = np.sqrt(np.mean((y_test_pred - y_test_true)**2))
mae = np.mean(np.abs(y_test_pred - y_test_true))

print("\nTest RMSE:", rmse)
print("Test MAE :", mae)


# %%
# =========================
# Plot training history
# =========================

plt.figure(figsize=(10,6))

plt.plot(history["train_total"], label="Train Total Loss")
plt.plot(history["train_data"], label="Train Data Loss")
plt.plot(history["train_pde"], label="Train PDE Loss")
plt.plot(history["val_data"], label="Validation Loss")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.title("PINN Training History")

plt.show()


# %%
# =========================
# FEM vs PINN comparison
# =========================

X_all = df[["Rdata","Zdata"]].values
X_all_scaled = x_scaler.transform(X_all)

X_all_t = torch.tensor(X_all_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    y_all_pred_scaled = model(X_all_t).cpu().numpy()

y_all_pred = y_scaler.inverse_transform(y_all_pred_scaled)

df_results = df.copy()
df_results["P_pred"] = y_all_pred

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(df_results["Zdata"], df_results["Rdata"],
            c=df_results["Pimag"], cmap="viridis")
plt.colorbar(label="FEM Pressure")

plt.title("FEM data")
plt.xlabel("Z")
plt.ylabel("R")

plt.subplot(1,2,2)
plt.scatter(df_results["Zdata"], df_results["Rdata"],
            c=df_results["P_pred"], cmap="viridis")
plt.colorbar(label="PINN Prediction")

plt.title("PINN prediction")
plt.xlabel("Z")
plt.ylabel("R")

plt.tight_layout()
plt.show()