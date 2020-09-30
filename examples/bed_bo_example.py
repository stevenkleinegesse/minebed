import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import minebed.networks as mn
import minebed.static.bed as bed

# --- DEFINE SIMULATOR ---- #


def simulator(d, p):

    y = p[:, 0] + p[:, 1] * d + np.random.normal(0, 1, size=len(params))
    return y.reshape(-1, 1)


# ----- HYPER-PARAMS ---- #

DATASIZE = 5000
BATCHSIZE = DATASIZE
N_EPOCH = 10000
BO_INIT_NUM = 5
BO_MAX_NUM = 5

# ----- PRIOR AND DOMAIN ----- #

mu = np.zeros(2)
sig = np.array([[9, 0], [0, 9]])
params = np.random.multivariate_normal(mu, sig, size=DATASIZE)

dom = [{
    'name': 'var_1',
    'type': 'continuous',
    'domain': (0, 10),
    'dimensionality': 1}]
con = None

# ---- DEFINE MODEL ----- #

net = mn.FullyConnected(var1_dim=2, var2_dim=1, L=1, H=50)
opt = optim.Adam(net.parameters(), lr=1e-3)
sch = StepLR(opt, step_size=1000, gamma=0.95)

# ----- TRAIN MODEL ------ #

bed_obj = bed.GradientFreeBED(
    model=net, optimizer=opt, scheduler=sch, simulator=simulator,
    prior=params, domain=dom, n_epoch=N_EPOCH, batch_size=BATCHSIZE,
    ma_window=100, constraints=con)

bed_obj.train(BO_init_num=BO_INIT_NUM, BO_max_num=BO_MAX_NUM, verbosity=True)
print('Optimal Design:')
print(bed_obj.d_opt)

print('Train Final Model')
bed_obj.train_final_model(n_epoch=20000, batch_size=DATASIZE)

bed_obj.bo_obj.plot_acquisition()
bed_obj.bo_obj.plot_convergence()

# ------ SAVING------- #
filename = './test'
bed_obj.save(filename)
