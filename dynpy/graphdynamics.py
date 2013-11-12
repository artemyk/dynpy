"""
Dynamics on graphs
"""

import numpy as np


import dynsys
class RandomWalker(dynsys.MultivariateDynamicalSystemBase):
	
	def __init__(self, graph, transMatrixClass = dynsys.DEFAULT_TRANSMX_CLASS):
		num_nodes = graph.shape[0]
		super(RandomWalker, self).__init__(num_nodes, transMatrixClass=transMatrixClass)
		graph = graph.astype('double')
		trans = graph / graph.sum(axis = 1)

		self.trans = self.transCls.finalizeMx( trans )
		self.denseTrans = self.transCls.toDense(self.trans)

		sMap = np.eye(num_nodes).astype('int')
		self.ndx2stateDict = {}
		self.state2ndxDict = {}
		for ndx, row in enumerate(sMap):
			state = tuple(row)
			self.ndx2stateDict[ndx] = state
			self.state2ndxDict[state] = ndx

		self.ndx2state = lambda ndx: self.ndx2stateDict[ndx]
		self.state2ndx = lambda stt: self.state2ndxDict[tuple(stt)]


	def iterateState(self, startState):
		return self.ndx2state( np.random.choice( self.num_nodes, None, replace=True, p=np.ravel( self.denseTrans[self.state2ndx(startState),:])) )

