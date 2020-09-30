import numpy as np
import random, os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributions import gamma

import minebed.networks as mn
import minebed.static.bed as bed

# --- DEFINE SIMULATOR ---- #

def sim_linear_torch(d, prior, device):

    # sample random normal noise
    n_n = torch.empty(
        (len(d), len(prior)),
        device=device,
        dtype=torch.float).normal_(mean=0, std=1)

    # sample random gamma noise
    n_g = gamma.Gamma(
        torch.tensor([2.0], device=device),
        torch.tensor([2.0], device=device)).sample(
            sample_shape=(len(d), len(prior))).reshape(len(d), len(prior))

    # perform forward pass
    y = (prior[:, 0] + torch.mul(prior[:, 1], d) + n_n + n_g).T
    ygrads = prior[:, 1].reshape(-1, 1)

    return y, ygrads


# ---- DEFINE SEED FUNCTION ---- #

def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----- SPECIFY CUDA DEVICE ---- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- HYPER-PARAMS ---- #

# seed
SEED = 12345678
seed_torch(SEED)

# Dimension of design variable in simulator
DIM = 1

# NN Size
L = 1
H = 10

# Sizes
DATASIZE = 10000
N_EPOCHS = 10000
BATCHSIZE = DATASIZE  # no batches works well

# Optimisation Params: Psi
LR_PSI = 1e-3
STEP_PSI = 1000
GAMMA_PSI = 1.0  # this would require fine-tuning

# Optimisation Params: Designs
LR_D = 1e-1
LR_D_init = LR_D
STEP_D = 1000
GAMMA_D = 1.0  # this would require fine-tuning

# ----- PRIOR AND DOMAIN ----- #

# Get regular prior samples
m, s = 0, 3
param_0 = np.random.normal(m, s, DATASIZE).reshape(-1, 1)
param_1 = np.random.normal(m, s, DATASIZE).reshape(-1, 1)
prior = np.hstack((param_0, param_1))

# Define bounds
bounds = [-10, 10]

# ---- INITIALIZE DESIGN AND MOVE TO DEVICE ---- #

# Initialise the design
# TODO: OTHER INITIALISATIONS?
d_init = np.random.uniform(bounds[0], bounds[-1], size=DIM).reshape(-1, 1)
print('Initial Design: ', d_init)

# Convert to PyTorch Tensors; put on CPU/GPU
d = torch.tensor(d_init, dtype=torch.float, device=device, requires_grad=False)
X = torch.tensor(prior, dtype=torch.float, device=device, requires_grad=False)
d.to(device)
X.to(device)

# ---- DEFINE MODEL ----- #

# define input dimensions
dim1, dim2 = prior.shape[-1], int(DIM)

# define model
model = mn.FullyConnected(var1_dim=dim1, var2_dim=dim2, L=L, H=H)
model.to(device)

# ----- DEFINE OPTIMIZERS AND SCHEDULERS ---- #

# AMSGrad Optimizer for NN params
optimizer_psi = optim.Adam(model.parameters(), lr=LR_PSI, amsgrad=True)

# AMSGrad Optimizer for designs
optimizer_design = optim.Adam([d], lr=LR_D, amsgrad=True)

# scheduler for NN params
scheduler_psi = StepLR(optimizer_psi, step_size=STEP_PSI, gamma=GAMMA_PSI)

# scheduler for designs
scheduler_design = StepLR(optimizer_design, step_size=STEP_D, gamma=GAMMA_D)

# ----- TRAIN MODEL ------ #

bed_sgd = bed.GradientBasedBED(
    model=model, optimizer=optimizer_psi, optimizer_design=optimizer_design,
    scheduler=scheduler_psi, scheduler_design=scheduler_design,
    simulator=sim_linear_torch, prior=X, n_epoch=N_EPOCHS,
    batch_size=BATCHSIZE, design_bounds=bounds, device=device, LB_type='NWJ')

print('Start Training:')
bed_sgd.train(d)
print('Optimal Design:', bed_sgd.designs[-1])

# ------ SAVING------- #
filename = './linear_sgd_test'
bed_sgd.save(filename)
