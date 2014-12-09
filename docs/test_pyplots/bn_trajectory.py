import matplotlib.pyplot as plt
import numpy as np
import dynpy

bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.budding_yeast_bn)

initState = np.zeros(bn.num_vars, 'uint8')
initState[ [1,3,6] ] = 1
plt.spy(bn.get_trajectory(start_state=initState, max_time=15))
plt.xlabel('Node')
plt.ylabel('Time')
