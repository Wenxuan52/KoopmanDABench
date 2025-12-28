import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

path_abs = r"C:\Users\chenc\CodeProject\Discrete CGKN\NSE"
device = "cuda:0"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

# Simulation Settings: Lt=1000, Lx=1, Ly=1, dt=0.001, dx=1/256, dy=1/256
Lt = 1000.
Lx = 1.
Ly = 1.

# Data Resolution: dt=0.01, dx=1/64, dy=1/64
u = np.load(path_abs + "/Data/NSE_Data(Noisy).npy")
t = np.arange(0, Lt, 0.01)
x = np.arange(0, Lx, 1/64)
y = np.arange(0, Ly, 1/64)
gridx, gridy = np.meshgrid(x, y, indexing="ij")


# Train / Test
Ntrain = 80000
Ntest = 20000
train_u_original = torch.from_numpy(u[:Ntrain]).to(torch.float32)
train_t = torch.from_numpy(t[:Ntrain]).to(torch.float32)
test_u_original = torch.from_numpy(u[-Ntest:]).to(torch.float32)
test_t = torch.from_numpy(t[-Ntest:]).to(torch.float32)

# Observed / Unobserved
indices_x_u1 = np.arange(0, 64, 8) # 8
indices_y_u1 = np.arange(0, 64, 8) # 8
indices_x_u2 = np.arange(0, 64) # 64
indices_y_u2 = np.arange(0, 64) # 64
indices_gridx_u1, indices_gridy_u1 = np.meshgrid(indices_x_u1, indices_y_u1, indexing="ij")
indices_gridx_u2, indices_gridy_u2 = np.meshgrid(indices_x_u2, indices_y_u2, indexing="ij")

train_u1_original = train_u_original[:, indices_gridx_u1, indices_gridy_u1]
train_u2_original = train_u_original[:, indices_gridx_u2, indices_gridy_u2]
test_u1_original = test_u_original[:, indices_gridx_u1, indices_gridy_u1]
test_u2_original = test_u_original[:, indices_gridx_u2, indices_gridy_u2]

class Normalizer:
    def __init__(self, x, eps=1e-9):
        # x is in the shape tensor (N, x, y)
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def decode(self, x):
        return x*(self.std.to(x.device) + self.eps) + self.mean.to(x.device)
normalizer = Normalizer(train_u_original)
train_u = normalizer.encode(train_u_original)
test_u = normalizer.encode(test_u_original)
train_u1 = train_u[:, indices_gridx_u1, indices_gridy_u1]
train_u2 = train_u[:, indices_gridx_u2, indices_gridy_u2]
test_u1 = test_u[:, indices_gridx_u1, indices_gridy_u1]
test_u2 = test_u[:, indices_gridx_u2, indices_gridy_u2]


############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        # x shape(t, x, y)
        x = x.unsqueeze(1)
        out = self.seq(x)
        out = out.squeeze(1)
        return out
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.ConvTranspose2d(1, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        # x shape(t, 20, 20)
        x = x.unsqueeze(1)
        out = self.seq(x)
        out = out.squeeze(1)
        return out
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z):
        super().__init__()
        self.dim_u1 = dim_u1
        self.f1 = nn.Parameter(1/dim_u1**0.5 * torch.rand(dim_u1, 1))
        self.g1 = nn.Parameter(1/(dim_u1*dim_z)**0.5 * torch.rand(dim_u1, dim_z))
        self.f2 = nn.Parameter(1/dim_z**0.5 * torch.rand(dim_z, 1))
        self.g2 = nn.Parameter(1/dim_z * torch.rand(dim_z, dim_z))
    def forward(self):
        return [self.f1, self.g1, self.f2, self.g2]
class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, u_extended):
        # Matrix Form Computation
        dim_u1 = self.cgn.dim_u1
        z = u_extended[:, dim_u1:]
        f1, g1, f2, g2 = self.cgn()
        z = z.unsqueeze(-1)
        u1_pred = f1 + g1@z
        z_pred = f2 + g2@z
        return torch.cat([u1_pred.squeeze(-1), z_pred.squeeze(-1)], dim=-1)


########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = len(indices_x_u1) * len(indices_y_u1)
dim_u2 = len(indices_x_u2) * len(indices_y_u2)
dim_z = 16*16

# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z
epochs = 1000
batch_size = 1000
train_tensor = torch.utils.data.TensorDataset(train_u[:-1], train_u[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_forecast_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []

autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    start_time = time.time()

    train_loss_forecast = 0.
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    for u, u_next in train_loader:
        u, u_next = u.to(device), u_next.to(device)
        # AutoEncoder
        u2 = u[:, indices_gridx_u2, indices_gridy_u2]
        z = cgkn.autoencoder.encoder(u2)
        u2_ae = cgkn.autoencoder.decoder(z)
        loss_ae = nnF.mse_loss(u2, u2_ae)

        #  State Forecast
        z_concat = z.reshape(-1, dim_z)
        u1 = u[:, indices_gridx_u1, indices_gridy_u1]
        u1_concat = u1.reshape(-1, dim_u1)
        u_extended = torch.cat([u1_concat, z_concat], dim=-1)
        u_extended_pred = cgkn(u_extended)
        u1_concat_pred = u_extended_pred[:, :dim_u1]
        z_concat_pred = u_extended_pred[:, dim_u1:]
        z_pred = z_concat_pred.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
        u1_pred = u1_concat_pred.reshape(-1, len(indices_x_u1), len(indices_y_u1))
        u2_pred = cgkn.autoencoder.decoder(z_pred)
        u1_next = u_next[:, indices_gridx_u1, indices_gridy_u1]
        u2_next = u_next[:, indices_gridx_u2, indices_gridy_u2]
        sse_u1 = nnF.mse_loss(u1_next, u1_pred, reduction="sum")
        sse_u2 = nnF.mse_loss(u2_next, u2_pred, reduction="sum")
        loss_forecast = (sse_u1 + sse_u2) / (u.shape[0]*(dim_u1+dim_u2))

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_pred)

        loss_total = loss_forecast + loss_ae + loss_forecast_z
        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_forecast += loss_forecast.item()
        train_loss_ae += loss_ae.item()
        train_loss_forecast_z += loss_forecast_z.item()
    train_loss_forecast /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_z /= train_num_batches
    train_loss_forecast_history.append(train_loss_forecast)
    train_loss_ae_history.append(train_loss_ae)
    train_loss_forecast_z_history.append(train_loss_forecast_z)

    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss fore:", round(train_loss_forecast, 4),
          " loss ae:", round(train_loss_ae, 4),
          " loss fore z:", round(train_loss_forecast_z, 4))
"""

# torch.save(cgkn, path_abs + r"NSE(Noisy)_CGKN_obs8_dimz16_stage1(Lambda)2.pt")
# np.save(path_abs + r"NSE(Noisy)_CGKN_obs8_dimz16_train_loss_forecast_history_stage1(Lambda)2.npy", train_loss_forecast_history)
# np.save(path_abs + r"NSE(Noisy)_CGKN_obs8_dimz16_train_loss_ae_history_stage1(Lambda)2.npy", train_loss_ae_history)
# np.save(path_abs + r"NSE(Noisy)_CGKN_obs8_dimz16_train_loss_forecast_z_history_stage1(Lambda)2.npy", train_loss_forecast_z_history)

cgkn = torch.load(path_abs + r"/Models/Model_CGKN/NSE(Noisy)_CGKN_obs8_dimz16_stage1.pt").to(device)


# # Model Diagnosis in Physical Space
# test_batch_size = 2000
# test_tensor = torch.utils.data.TensorDataset(test_u)
# test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
# test_u1_pred = torch.zeros_like(test_u1)
# test_u2_pred = torch.zeros_like(test_u2)
# si = 0
# for u in test_loader:
#     test_u_batch = u[0].to(device)
#     test_u1_batch = test_u_batch[:, indices_gridx_u1, indices_gridy_u1]
#     test_u2_batch = test_u_batch[:, indices_gridx_u2, indices_gridy_u2]
#     test_u1_concat_batch = test_u1_batch.reshape(-1, dim_u1)
#     with torch.no_grad():
#         test_z_batch = cgkn.autoencoder.encoder(test_u2_batch)
#         test_z_concat_batch = test_z_batch.reshape(-1, dim_z)
#         test_u_extended_batch = torch.cat([test_u1_concat_batch, test_z_concat_batch], dim=-1)
#         test_u_extended_pred_batch = cgkn(test_u_extended_batch )
#         test_u1_concat_pred_batch = test_u_extended_pred_batch[:, :dim_u1]
#         test_z_concat_pred_batch = test_u_extended_pred_batch[:, dim_u1:]
#         test_u1_pred_batch = test_u1_concat_pred_batch.reshape(-1, len(indices_x_u1), len(indices_y_u1))
#         test_u2_pred_batch = cgkn.autoencoder.decoder(test_z_concat_pred_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5) ))
#     ei = si + test_batch_size
#     test_u1_pred[si:ei] = test_u1_pred_batch
#     test_u2_pred[si:ei] = test_u2_pred_batch
#     si = ei
# print(nnF.mse_loss(test_u2[1:], test_u2_pred[:-1]))
# test_u2_original_pred = normalizer.decode(test_u2_pred)
# print(nnF.mse_loss(test_u2_original[1:], test_u2_original_pred[:-1]))



#################################################################
################# Noise Coefficient & CGFilter  #################
#################################################################
train_batch_size = 2000
train_tensor = torch.utils.data.TensorDataset(train_u)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=False, batch_size=train_batch_size)
train_u_extended = torch.zeros(Ntrain, dim_u1+dim_z)
train_u_extended_pred = torch.zeros(Ntrain, dim_u1+dim_z)
si = 0
for u in train_loader:
    train_u_batch = u[0].to(device)
    train_u1_batch = train_u_batch[:, indices_gridx_u1, indices_gridy_u1]
    train_u2_batch = train_u_batch[:, indices_gridx_u2, indices_gridy_u2]
    train_u1_concat_batch = train_u1_batch.reshape(-1, dim_u1)
    with torch.no_grad():
        train_z_batch = cgkn.autoencoder.encoder(train_u2_batch)
        train_z_concat_batch = train_z_batch.reshape(-1, dim_z)
        train_u_extended_batch = torch.cat([train_u1_concat_batch, train_z_concat_batch], dim=-1)
        train_u_extended_pred_batch = cgkn(train_u_extended_batch)
    ei = si + train_batch_size
    train_u_extended[si:ei] = train_u_extended_batch
    train_u_extended_pred[si:ei] = train_u_extended_pred_batch
    si = ei
sigma_hat = torch.sqrt( torch.mean( (train_u_extended[1:] - train_u_extended_pred[:-1])**2, dim=0 ) )
sigma_hat[dim_u1:] = 0.5 # sigma_z is set manually


def CGFilter(cgkn, sigma, u1, mu0, R0):
    # u1: (t, x, 1)
    # mu0: (x, 1)
    # R0: (x, x)
    device = u1.device
    Nt = u1.shape[0]
    dim_u1 = u1.shape[1]
    dim_z = mu0.shape[0]
    s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
    mu_pred = torch.zeros((Nt, dim_z, 1)).to(device)
    R_pred = torch.zeros((Nt, dim_z, dim_z)).to(device)
    f1, g1, f2, g2 = cgkn.cgn()
    for n in range(Nt):
        mu1 = f2 + g2@mu0 + g2@R0@g1.T@ torch.inverse(s1@s1.T+g1@R0@g1.T)@(u1[n]-f1-g1@mu0)
        R1 = g2@R0@g2.T + s2@s2.T - g2@R0@g1.T@torch.inverse(s1@s1.T+g1@R0@g1.T)@g1@R0@g2.T
        mu_pred[n] = mu1
        R_pred[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_pred, R_pred)

########################################################
################# Train cgkn (Stage2)  #################
########################################################

# Stage 2: Train cgkn with loss_forecast + loss_da + loss_ae + loss_forecast_z
short_steps = 2
long_steps = 2000
cut_point = 500

Niters = 50000
train_batch_size = 1000
train_loss_forecast_history = []
train_loss_da_history = []
train_loss_ae_history = []
train_loss_forecast_z_history = []
# Re-initialize Model
autoencoder = AutoEncoder().to(device)
cgn = CGN(dim_u1, dim_z).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for itr in range(1, Niters+1):
    start_time = time.time()

    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u_short = torch.stack([train_u[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device)
    # AutoEncoder
    u2_short = u_short[:, :, indices_gridx_u2, indices_gridy_u2]
    z_short = cgkn.autoencoder.encoder( u2_short.reshape(-1, len(indices_x_u2), len(indices_y_u2)) ).reshape(short_steps, train_batch_size, int(dim_z**0.5), int(dim_z**0.5))
    u2_ae_short = cgkn.autoencoder.decoder(z_short.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))).reshape(short_steps, train_batch_size, len(indices_x_u2), len(indices_y_u2))
    loss_ae = nnF.mse_loss(u2_short, u2_ae_short)

    # State Prediction
    u1_short = u_short[:, :, indices_gridx_u1, indices_gridy_u1]
    u1_concat_short = u1_short.reshape(short_steps, train_batch_size, dim_u1)
    z_concat_short = z_short.reshape(short_steps, train_batch_size, dim_z)
    u_extended0_short = torch.cat([u1_concat_short[0], z_concat_short[0]], dim=-1)
    u_extended_pred_short = torch.zeros(short_steps, train_batch_size, dim_u1+dim_z).to(device)
    u_extended_pred_short[0] = u_extended0_short
    for n in range(1, short_steps):
        u_extended1_short = cgkn(u_extended0_short)
        u_extended_pred_short[n] = u_extended1_short
        u_extended0_short = u_extended1_short
    z_concat_pred_short = u_extended_pred_short[:, :, dim_u1:]
    loss_forecast_z = nnF.mse_loss(z_concat_short[1:], z_concat_pred_short[1:])

    u1_concat_pred_short = u_extended_pred_short[:, :, :dim_u1]
    u1_pred_short = u1_concat_pred_short.reshape(short_steps, train_batch_size, len(indices_x_u1), len(indices_y_u1))
    z_pred_short = z_concat_pred_short.reshape(short_steps, train_batch_size, int(dim_z ** 0.5), int(dim_z ** 0.5))
    u2_pred_short = cgkn.autoencoder.decoder(z_pred_short.reshape(-1, int(dim_z ** 0.5), int(dim_z ** 0.5))).reshape(short_steps, train_batch_size, len(indices_x_u2), len(indices_y_u2))
    sse_u1_short = nnF.mse_loss(u1_short[1:], u1_pred_short[1:], reduction="sum")
    sse_u2_short = nnF.mse_loss(u2_short[1:], u2_pred_short[1:], reduction="sum")
    loss_forecast = (sse_u1_short + sse_u2_short) / ((short_steps-1)*train_batch_size*(dim_u1+dim_u2))

    # DA
    head_idx_long = torch.from_numpy( np.random.choice(Ntrain-long_steps+1, size=1, replace=False) )
    u_long = train_u[head_idx_long:head_idx_long + long_steps].to(device)
    u1_long = u_long[:, indices_gridx_u1, indices_gridy_u1]
    u2_long = u_long[:, indices_gridx_u2, indices_gridy_u2]
    u1_concat_long = u1_long.reshape(-1, dim_u1)
    mu_z_concat_pred_long = CGFilter(cgkn, sigma_hat.to(device), u1_concat_long.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.01*torch.eye(dim_z).to(device))[0].squeeze(-1)
    mu_z_pred_long = mu_z_concat_pred_long.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
    mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long[cut_point:])
    loss_da = nnF.mse_loss(u2_long[cut_point:], mu_pred_long)

    loss_total = loss_forecast + loss_da + loss_ae + loss_forecast_z
    if torch.isnan(loss_total):
        print(itr, "nan")
        continue

    # print(torch.cuda.memory_allocated(device=device) / 1024**2)
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    train_loss_forecast_history.append(loss_forecast.item())
    train_loss_da_history.append(loss_da.item())
    train_loss_ae_history.append(loss_ae.item())
    train_loss_forecast_z_history.append(loss_forecast_z.item())

    end_time = time.time()
    print("itr", itr,
          " time:", round(end_time-start_time, 4),
          " loss fore:", round(loss_forecast.item(), 4),
          " loss da:", round(loss_da.item(),4),
          " loss ae:", round(loss_ae.item(),4),
          " loss fore z:", round(loss_forecast_z.item(), 4))
"""

# torch.save(cgkn, path_abs + r"NSE(Noisy)_CGKN_obs8_dimz16_stage2(FromScratch)(Lambda)2.pt")
# np.save(path_abs + r"NSE_CGKN(Noisy)_obs8_dimz16_train_loss_forecast_history_stage2(FromScratch)(Lambda)2.npy", train_loss_forecast_history)
# np.save(path_abs + r"NSE_CGKN(Noisy)_obs8_dimz16_train_loss_ae_history_stage2(FromScratch)(Lambda)2.npy", train_loss_ae_history)
# np.save(path_abs + r"NSE_CGKN(Noisy)_obs8_dimz16_train_loss_forecast_z_history_stage2(FromScratch)(Lambda)2.npy", train_loss_forecast_z_history)
# np.save(path_abs + r"NSE_CGKN(Noisy)_obs8_dimz16_train_loss_da_history_stage2(FromScratch)(Lambda)2.npy", train_loss_da_history)

cgkn = torch.load(path_abs + r"/Models/Model_CGKN/NSE(Noisy)_CGKN_obs8_dimz16_stage2.pt").to(device)


#####################################################################################
################# DA Uncertainty Quantification via Residual Analysis ###############
#####################################################################################
# Data Assimilation of Train Data
batch_steps = 1000
train_mu_pred = torch.zeros(Ntrain, int(dim_u2**0.5), int(dim_u2**0.5)).to(device)
train_mu_z0 = torch.zeros(dim_z, 1).to(device)
train_R_z0 = 0.01 * torch.eye(dim_z).to(device)
for si in np.arange(0, Ntrain, batch_steps):
    with torch.no_grad():
        train_mu_z_concat_pred_batch, train_mu_R_pred_batch = CGFilter(cgkn,
                                                                     sigma_hat.to(device),
                                                                     train_u1.reshape(-1, dim_u1).unsqueeze(-1)[si:si+batch_steps].to(device),
                                                                     train_mu_z0,
                                                                     train_R_z0)
        train_mu_z_pred_batch = train_mu_z_concat_pred_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
        train_mu_pred_batch = cgkn.autoencoder.decoder(train_mu_z_pred_batch)
    train_mu_pred[si:si+batch_steps] = train_mu_pred_batch
    train_mu_z0 = train_mu_z_concat_pred_batch[-1]
    train_R_z0 = train_mu_R_pred_batch[-1]
train_mu_pred = train_mu_pred.to("cpu")
print(nnF.mse_loss(train_u2[cut_point:], train_mu_pred[cut_point:]).item())
train_mu_original_pred = normalizer.decode(train_mu_pred)
print(nnF.mse_loss(train_u2_original[cut_point:], train_mu_original_pred[cut_point:]).item())

# Target Variable: Residual (std of posterior mean)
train_mu_original_std = torch.abs(train_u2_original[cut_point:] - train_mu_original_pred[cut_point:])

class UncertaintyNet(nn.Module):
    def __init__(self, dim_u1, dim_u2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_u1, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, dim_u2))

    def forward(self, x):
        out = self.net(x)
        return out

epochs = 5000
train_batch_size = 2000
train_tensor = torch.utils.data.TensorDataset(train_u1[cut_point:], train_mu_original_std)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_uncertainty_history = []

uncertainty_net = UncertaintyNet(dim_u1, dim_u2).to(device)
optimizer = torch.optim.Adam(uncertainty_net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss_uncertainty = 0.
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = uncertainty_net(x.reshape(x.shape[0], dim_u1))
        loss = nnF.mse_loss(y.reshape(y.shape[0], dim_u2), out)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss_uncertainty += loss.item()
    train_loss_uncertainty /= train_num_batches
    train_loss_uncertainty_history.append(train_loss_uncertainty)
    end_time = time.time()
    print("ep", ep,
          " time:", round(end_time - start_time, 4),
          " loss uncertainty:", round(train_loss_uncertainty, 4))

# torch.save(uncertainty_net, path_abs + r"/Models/Model_CGKN/NSE_UQNet_obs8_dimz16.pt")
# np.save(path_abs + r"/Models/Model_CGKN/NSE_UQNet_obs8_dimz16_train_loss_uncertainty_history.npy", train_loss_uncertainty_history)

uncertainty_net = torch.load(path_abs + r"/Models/Model_CGKN/NSE_UQNet_obs8_dimz16.pt").to(device)


#############################################
################# Test cgkn #################
#############################################
cgkn.cpu()
uncertainty_net.cpu()
device = next(cgkn.parameters()).device

# CGKN for One-Step Prediction
test_batch_size = 1000
test_tensor = torch.utils.data.TensorDataset(test_u)
test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
test_u1_pred = torch.zeros_like(test_u1)
test_u2_pred = torch.zeros_like(test_u2)
si = 0
for u in test_loader:
    test_u_batch = u[0].to(device)
    test_u1_batch = test_u_batch[:, indices_gridx_u1, indices_gridy_u1]
    test_u2_batch = test_u_batch[:, indices_gridx_u2, indices_gridy_u2]
    test_u1_concat_batch = test_u1_batch.reshape(-1, dim_u1)
    with torch.no_grad():
        test_z_batch = cgkn.autoencoder.encoder(test_u2_batch)
        test_z_concat_batch = test_z_batch.reshape(-1, dim_z)
        test_u_extended_batch = torch.cat([test_u1_concat_batch, test_z_concat_batch], dim=-1)
        test_u_extended_pred_batch = cgkn(test_u_extended_batch )
        test_u1_concat_pred_batch = test_u_extended_pred_batch[:, :dim_u1]
        test_z_concat_pred_batch = test_u_extended_pred_batch[:, dim_u1:]
        test_u1_pred_batch = test_u1_concat_pred_batch.reshape(-1, len(indices_x_u1), len(indices_y_u1))
        test_u2_pred_batch = cgkn.autoencoder.decoder(test_z_concat_pred_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5) ))
    ei = si + test_batch_size
    test_u1_pred[si:ei] = test_u1_pred_batch
    test_u2_pred[si:ei] = test_u2_pred_batch
    si = ei
print(nnF.mse_loss(test_u2[1:], test_u2_pred[:-1]).item())
test_u2_original_pred = normalizer.decode(test_u2_pred)
print(nnF.mse_loss(test_u2_original[1:], test_u2_original_pred[:-1]).item())


# CGKN for Data Assimilation
st = time.time()
batch_steps = 1000
test_mu_pred = torch.zeros(Ntest, int(dim_u2**0.5), int(dim_u2**0.5)).to(device)
test_mu_z0 = torch.zeros(dim_z, 1).to(device)
test_R_z0 = 0.01 * torch.eye(dim_z).to(device)
for si in np.arange(0, Ntest, batch_steps):
    with torch.no_grad():
        test_mu_z_concat_pred_batch, test_mu_R_pred_batch = CGFilter(cgkn,
                                                                     sigma_hat.to(device),
                                                                     test_u1.reshape(-1, dim_u1).unsqueeze(-1)[si:si+batch_steps].to(device),
                                                                     test_mu_z0,
                                                                     test_R_z0)
        test_mu_z_pred_batch = test_mu_z_concat_pred_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
        test_mu_pred_batch = cgkn.autoencoder.decoder(test_mu_z_pred_batch)
    test_mu_pred[si:si+batch_steps] = test_mu_pred_batch
    test_mu_z0 = test_mu_z_concat_pred_batch[-1]
    test_R_z0 = test_mu_R_pred_batch[-1]
test_mu_pred = test_mu_pred.to("cpu")
print(nnF.mse_loss(test_u2[cut_point:], test_mu_pred[cut_point:]).item())
test_mu_original_pred = normalizer.decode(test_mu_pred)
print(nnF.mse_loss(test_u2_original[cut_point:], test_mu_original_pred[cut_point:]).item())

# uncertainty_net for Uncertainty Quantification
with torch.no_grad():
    test_mu_original_std_pred = uncertainty_net(test_u[:, indices_gridx_u1, indices_gridy_u1].to(device).reshape(-1, dim_u1)).cpu().reshape(-1, int(dim_u2**0.5), int(dim_u2**0.5)).cpu()
et = time.time()
time_DA = et - st
print("DA time:", time_DA)
# np.savez(path_abs + "/Data/NSE(Noisy)_CGKN_DA.npz", mean=test_mu_original_pred, std=test_mu_original_std_pred)



# CGKN for Multi-Step State Prediction (Advanced)
test_short_steps = 10
mask = torch.ones(Ntest, dtype=torch.bool)
mask[::test_short_steps] = False
test_u_initial = test_u[::test_short_steps]
test_batch_size = 100
test_tensor = torch.utils.data.TensorDataset(test_u_initial)
test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
test_u2_shortPred = torch.zeros(test_u_initial.shape[0] * test_short_steps, 64, 64)
si = 0
for u in test_loader:
    test_u_initial_batch = u[0].to(device)
    test_u1_initial_batch = test_u_initial_batch[:, indices_gridx_u1, indices_gridy_u1]
    test_u2_initial_batch = test_u_initial_batch[:, indices_gridx_u2, indices_gridy_u2]
    test_u1_initial_concat_batch = test_u1_initial_batch.reshape(test_u1_initial_batch.shape[0], dim_u1)
    with torch.no_grad():
        test_z_initial_batch = cgkn.autoencoder.encoder(test_u2_initial_batch)
    test_z_initial_concat_batch = test_z_initial_batch.reshape(test_z_initial_batch.shape[0], dim_z)
    test_u_extended_initial_batch = torch.cat([test_u1_initial_concat_batch, test_z_initial_concat_batch], dim=-1)
    test_u2_shortPred_batch = torch.zeros(test_short_steps, test_u_initial_batch.shape[0], int(dim_u2**0.5), int(dim_u2**0.5)).to(device)
    test_u2_shortPred_batch[0] = test_u2_initial_batch
    with torch.no_grad():
        for n in range(test_short_steps-1):
            test_u_extended_next_batch = cgkn(test_u_extended_initial_batch)
            test_u1_concat_next_batch = test_u_extended_next_batch[:, :dim_u1]
            test_z_concat_next_batch = test_u_extended_next_batch[:, dim_u1:]
            test_z_next_batch = test_z_concat_next_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
            test_u2_next_batch = cgkn.autoencoder.decoder(test_z_next_batch)
            test_u2_shortPred_batch[n+1] = test_u2_next_batch
            test_z_next_ae_batch = cgkn.autoencoder.encoder(test_u2_next_batch)
            test_z_concat_next_ae_batch = test_z_next_ae_batch.reshape(-1, dim_z)
            test_u_extended_initial_batch = torch.cat([test_u1_concat_next_batch, test_z_concat_next_ae_batch], dim=-1)
    test_u2_shortPred_batch = test_u2_shortPred_batch.permute(1, 0, 2, 3).reshape(-1, int(dim_u2**0.5), int(dim_u2**0.5))
    test_u2_shortPred[si:si+test_u2_shortPred_batch.shape[0]] = test_u2_shortPred_batch
    si += test_u2_shortPred_batch.shape[0]
test_u2_shortPred = test_u2_shortPred[:Ntest]
print(nnF.mse_loss(test_u2[mask], test_u2_shortPred[mask]).item())
test_u2_original_shortPred = normalizer.decode(test_u2_shortPred)
print(nnF.mse_loss(test_u2_original[mask], test_u2_original_shortPred[mask]).item())



# CGKN: Number of Parameters
len( torch.nn.utils.parameters_to_vector( cgkn.cgn.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.encoder.parameters() ) )
len( torch.nn.utils.parameters_to_vector( cgkn.autoencoder.decoder.parameters() ) )





# # CGKN for Multi-Step State Prediction (Naive)
# test_short_steps = 10
# mask = torch.ones(Ntest, dtype=torch.bool)
# mask[::test_short_steps] = False
# test_u_initial = test_u[::test_short_steps]
# test_batch_size = 100
# test_tensor = torch.utils.data.TensorDataset(test_u_initial)
# test_loader = torch.utils.data.DataLoader(test_tensor, shuffle=False, batch_size=test_batch_size)
# test_u2_shortPred = torch.zeros(test_u_initial.shape[0] * test_short_steps, 64, 64)
# si = 0
# for u in test_loader:
#     test_u_initial_batch = u[0].to(device)
#     test_u1_initial_batch = test_u_initial_batch[:, indices_gridx_u1, indices_gridy_u1]
#     test_u2_initial_batch = test_u_initial_batch[:, indices_gridx_u2, indices_gridy_u2]
#     test_u1_initial_concat_batch = test_u1_initial_batch.reshape(test_u1_initial_batch.shape[0], dim_u1)
#     with torch.no_grad():
#         test_z_initial_batch = cgkn.autoencoder.encoder(test_u2_initial_batch)
#     test_z_initial_concat_batch = test_z_initial_batch.reshape(test_z_initial_batch.shape[0], dim_z)
#     test_u_extended_initial_batch = torch.cat([test_u1_initial_concat_batch, test_z_initial_concat_batch], dim=-1)
#     test_u_extended_shortPred_batch = torch.zeros(test_short_steps, test_u_initial_batch.shape[0], dim_u1+dim_z).to(device) # (t, N, x)
#     test_u_extended_shortPred_batch[0] = test_u_extended_initial_batch
#     with torch.no_grad():
#         for n in range(test_short_steps-1):
#             test_u_extended_shortPred_batch[n+1] = cgkn(test_u_extended_shortPred_batch[n])
#     test_u1_concat_shortPred_batch = test_u_extended_shortPred_batch[:, :, :dim_u1]
#     test_z_concat_shortPred_batch = test_u_extended_shortPred_batch[:, :, dim_u1:]
#     test_u1_shortPred_batch = test_u1_concat_shortPred_batch.reshape(test_short_steps, test_u1_concat_shortPred_batch.shape[1], int(dim_u1**0.5), int(dim_u1**0.5))
#     test_z_shortPred_batch = test_z_concat_shortPred_batch.reshape(test_short_steps, test_z_concat_shortPred_batch.shape[1], int(dim_z**0.5), int(dim_z**0.5))
#     with torch.no_grad():
#         test_u2_shortPred_batch = cgkn.autoencoder.decoder( test_z_shortPred_batch.reshape(-1, int(dim_z**0.5), int(dim_z**0.5)) ).reshape(test_short_steps, test_z_shortPred_batch.shape[1], int(dim_u2**0.5), int(dim_u2**0.5))
#     test_u2_shortPred_batch = test_u2_shortPred_batch.permute(1, 0, 2, 3).reshape(-1, int(dim_u2**0.5), int(dim_u2**0.5))
#     test_u2_shortPred[si:si+test_u2_shortPred_batch.shape[0]] = test_u2_shortPred_batch
#     si = si + test_u2_shortPred_batch.shape[0]
# test_u2_shortPred = test_u2_shortPred[:Ntest]
# nnF.mse_loss(test_u2[mask], test_u2_shortPred[mask])
# test_u2_original_shortPred = normalizer.decode(test_u2_shortPred)
# nnF.mse_loss(test_u2_original[mask], test_u2_original_shortPred[mask])


# CGKN for Multi-Step State Prediction (Advanced) (Data start from 950)
si = 15000
steps = 11
u_initial = test_u[[si]].to(device)
u1_initial = u_initial[:, indices_gridx_u1, indices_gridy_u1]
u2_initial = u_initial[:, indices_gridx_u2, indices_gridy_u2]
u1_concat_initial = u1_initial.reshape(-1, dim_u1)
with torch.no_grad():
    z_initial = cgkn.autoencoder.encoder(u2_initial)
z_concat_initial = z_initial.reshape(-1, dim_z)
u_extended_initial = torch.cat([u1_concat_initial, z_concat_initial], dim=-1)
u2_shortPred = torch.zeros(steps, 1, 64, 64)
u2_shortPred[0] = u2_initial
with torch.no_grad():
    for n in range(steps-1):
        u_extended_next = cgkn(u_extended_initial)
        u1_concat_next = u_extended_next[:, :dim_u1]
        z_concat_next = u_extended_next[:, dim_u1:]
        z_next = z_concat_next.reshape(-1, int(dim_z**0.5), int(dim_z**0.5))
        u2_next = cgkn.autoencoder.decoder(z_next)
        u2_shortPred[n+1] = u2_next
        z_next_ae = cgkn.autoencoder.encoder(u2_next)
        z_concat_next_ae = z_next_ae.reshape(-1, dim_z)
        u_extended_initial = torch.cat([u1_concat_next, z_concat_next_ae], dim=-1)
u2_shortPred = u2_shortPred.permute(1, 0, 2, 3).reshape(-1, int(dim_u2**0.5), int(dim_u2**0.5))
u2_original_shortPred = normalizer.decode(u2_shortPred)
# np.save(path_abs + r"/Data/NSE(Noisy)_CGKN_SF_950initial_10steps.npy", u2_original_shortPred)

