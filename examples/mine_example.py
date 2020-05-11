import numpy as np
import minebed.networks as mn
import minebed.mine as mine

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


# --- SAMPLE MOCK DATA ----- #
DATASIZE = 20000
rho = 0.9
samples = np.random.multivariate_normal(
    np.zeros(2), np.array([[1, rho], [rho, 1]]), size=DATASIZE)

X = samples[:, 0].reshape(-1, 1)
Y = samples[:, 1].reshape(-1, 1)
train_data = (X, Y)

mi_an = -0.5 * np.log(1 - rho**2)
print(mi_an)

model = mn.FullyConnected(var1_dim=1, var2_dim=1, L=1, H=[10])
mine_obj = mine.MINE(model, train_data, lr=5 * 1e-4)

mine_obj.train(n_epoch=10000, batch_size=DATASIZE)

print(np.mean(mine_obj.train_lb[-100:]))
