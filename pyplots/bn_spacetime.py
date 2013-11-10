import numpy as np
import matplotlib.pyplot as plt

import dynpy.bn
import dynpy.sample_bn_nets

num_steps = 15
bn = dynpy.bn.BooleanNetworkFromTruthTables(rules=dynpy.sample_bn_nets.yeast)

spacetime = np.zeros(shape=(num_steps,bn.num_nodes), dtype='int')

cState = np.zeros(bn.num_nodes)
cState[ [1,3,6] ] = 1.0

for i in range(num_steps):
    spacetime[i,:] = cState
    cState = bn.getNextState(cState)

plt.spy(spacetime)