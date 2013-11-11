"""
Dynamics on graphs
"""

class RandomWalker(object):
	
	def __init__(self, graph):

		self.trans = graph / graph.sum(axis = 1)