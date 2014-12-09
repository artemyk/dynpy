import matplotlib ; matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import dynpy

ca = dynpy.ca.CellularAutomaton(num_vars=100, num_neighbors=1, ca_rule_number=110)

initState = np.zeros(ca.num_vars, 'uint8')
initState[int(ca.num_vars/2)] = 1
plt.spy(ca.get_trajectory(start_state=initState, max_time=50))
plt.xlabel('Node')
plt.ylabel('Time')
