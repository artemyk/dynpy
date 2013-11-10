import numpy as np
import matplotlib.pyplot as plt

import dynpy.ca

num_nodes = 100
num_steps = 50
ca = dynpy.ca.CellularAutomaton(num_nodes=num_nodes, num_neighbors=1, ca_rule_number=110)
spacetime = np.zeros(shape=(num_steps,num_nodes), dtype='int')

cState = np.zeros(num_nodes)
cState[int(num_nodes/2)] = 1.0

for i in range(num_steps):
    spacetime[i,:] = cState
    cState = ca.getNextState(cState)

plt.spy(spacetime)