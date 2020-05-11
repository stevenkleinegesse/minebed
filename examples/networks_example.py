# NOTE: THIS IS TEMPORARY ONLY!
import sys
sys.path.append('/Users/steven/phd/repos/minebed/src/')

import minebed.networks as mn

var1_dim, var2_dim = 2, 1

L = 1
H = 10
model = mn.FullyConnected(var1_dim, var2_dim, L, H)
print(model)

L = 3
H = [10, 20, 30]
model = mn.FullyConnected(var1_dim, var2_dim, L, H)
print(model)
print(model.layers)

# NEED TO FIGURE OUT HOW TO WRITE PROPER TESTS FOR PYTEST