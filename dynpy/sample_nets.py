"""This module provides several sample networks and dynamical systems for use
in testing and development.

Right now it includes:

``yeast_cellcycle_bn``: A 11-node yeast-cell cycle Boolean network.  It is in
the 'truthtable' format. For more details, see Li et al, The yeast cell-cycle 
network is robustly designed, PNAS, 2004.

``karateclub_net``: A 34-node graph representing Zachary's karate-club network.
It is in numpy array format.

``test_bn``: A simple 4-node BN for testing

``test2_bn``: Another simple 4-node BN for testing

"""

import numpy as np

# "The yeast cell-cycle network is robustly designed",
# Fangting Li, Tao Long, Ying Lu, Qi Ouyang, Chao Tang,
#  PNAS  April 6, 2004,  vol. 101  no. 14  4781-4786.  
yeast_cellcycle_bn = [None]*11
yeast_cellcycle_bn[0] = ['Cln3',  ['Cln3'], [0, 0]]
yeast_cellcycle_bn[1] = ['MBF',
                         ['Cln3', 'Clb1,2', 'MBF'],
                         [1, 0, 1, 1, 0, 0, 1, 0]]
yeast_cellcycle_bn[2] = ['SBF',
                         ['Cln3', 'Clb1,2', 'SBF'],
                         [1, 0, 1, 1, 0, 0, 1, 0]]
yeast_cellcycle_bn[3] = ['Cln1,2',['SBF', 'Cln1,2'], [1, 1, 0, 0]]
yeast_cellcycle_bn[4] = ['Sic1',
                         ['Cln1,2', 'Clb5,6', 'Clb1,2', 'Cdc20', 'Sic1'],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]]
yeast_cellcycle_bn[5] = ['Swi5',
                         ['Cdc20', 'Clb1,2', 'Mcm1', 'Swi5'],
                         [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]]
yeast_cellcycle_bn[6] = ['Cdc20',
                         ['Clb1,2', 'Mcm1', 'Cdc20'],
                         [1, 1, 1, 1, 1, 1, 0, 0]]
yeast_cellcycle_bn[7] = ['Clb5,6',
                         ['MBF', 'Cdc20', 'Cdh1', 'Clb5,6'],
                         [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]]
yeast_cellcycle_bn[8] = ['Cdh1',
                         ['Cln1,2','Swi5','Cdc20','Clb5,6','Clb1,2','Cdh1'],
                         [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                          1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                          1, 1, 0, 0, 0, 0, 0, 0, 1, 0]]
yeast_cellcycle_bn[9] = ['Clb1,2',
                         ['Sic1', 'Cdc20', 'Clb5,6', 'Cdh1', 'Mcm1', 'Clb1,2'],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                          0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                          1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1,
                          1, 1, 1, 0, 0, 0, 1, 1, 1, 0]]
yeast_cellcycle_bn[10]= ['Mcm1',
                         ['Clb5,6', 'Clb1,2', 'Mcm1'],
                         [1, 1, 1, 1, 1, 1, 0, 0]]


# "From Genes to Flower Patterns and Evolution: Dynamic Models of Gene Regulatory Networks"
# A. Chaos, M. Aldana, C. Espinosa-Soto, B. G. P. de Leon, A. G. Arroyo, E. R. Alvarez-Buylla,
# Journal of Plant Growth Regulation, vol. 25, n. 4, 2006, pp. 278-289
arabidopsis_bn = [
  ['AP3', ['AP1', 'LFY', 'AG', 'PI', 'SEP', 'AP3', 'UFO'], lambda AP1, LFY, AG, PI, SEP, AP3, UFO: ((AG and PI and SEP and AP3) or (AP1 and PI and SEP and AP3) or (LFY and UFO))],
  ['UFO', ['UFO'], lambda UFO: ((UFO))],
  ['FUL', ['AP1', 'TFL1'], lambda AP1, TFL1: ((not AP1 and not TFL1))],
  ['FT', ['EMF1'], lambda EMF1: ((not EMF1))],
  ['AP1', ['FT', 'LFY', 'AG', 'TFL1'], lambda FT, LFY, AG, TFL1: ((not AG and not TFL1) or (LFY and not AG) or (FT and not AG))],
  ['EMF1', ['LFY'], lambda LFY: ((not LFY))],
  ['LFY', ['FUL', 'AP1', 'EMF1', 'TFL1'], lambda FUL, AP1, EMF1, TFL1: ((not TFL1) or (not EMF1))],
  ['AP2', ['TFL1'], lambda TFL1: ((not TFL1))],
  ['WUS', ['WUS', 'AG', 'SEP'], lambda WUS, AG, SEP: ((WUS and not SEP) or (WUS and not AG))],
  ['AG', ['AP1', 'LFY', 'AP2', 'WUS', 'AG', 'LUG', 'CLF', 'TFL1', 'SEP'], lambda AP1, LFY, AP2, WUS, AG, LUG, CLF, TFL1, SEP: ((LFY and AG and SEP) or (not AP2 and not TFL1) or (LFY and not CLF) or (LFY and not LUG) or (LFY and WUS) or (not AP1 and LFY) or (LFY and not AP2))],
  ['LUG', ['LUG'], lambda LUG: LUG],
  ['CLF', ['CLF'], lambda CLF: CLF],
  ['TFL1', ['AP1', 'EMF1', 'LFY', 'AP2'], lambda AP1, EMF1, LFY, AP2: ((not AP1 and EMF1 and not LFY))],
  ['PI', ['AP1', 'LFY', 'AG', 'PI', 'SEP', 'AP3'], lambda AP1, LFY, AG, PI, SEP, AP3: ((AP1 and PI and SEP and AP3) or (AG and PI and SEP and AP3) or (LFY and AG) or (LFY and AP3))],
  ['SEP', ['LFY'], lambda LFY: ((LFY))]
]

# "The topology of the regulatory interactions predicts the expression pattern of the 
# segment polarity genes in Drosophila melanogaster", R. Albert and H. G. Othmer,
# Journal of Theoretical Biology, 2003, vol. 223, no. 1, pp. 1-18.
# ....

karateclub_net = np.array([
	[0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
	[1,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
	[1,1,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0],
	[1,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1],
	[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
	[1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
	[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
	[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
	[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,1],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
	[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1],
	[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1],
	[0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
	[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1],
	[0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,0,0,1,1,1,0,1],
	[0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,0]
 ])



test_bn = [
  ['Node1',
    [ 'Node1','Node2','Node3','Node4' ] ,
    [ 1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   0, 0, 0, 0]],
  ['Node2',
    [ 'Node1','Node2','Node3','Node4' ] ,
    [ 1, 1, 1, 1,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]],
  ['Node3',
    [ 'Node1','Node2','Node3','Node4' ] ,
    [ 1, 1, 1, 0,   1, 0, 0, 0,   1, 1, 1, 0,   1, 0, 0, 0]],
  ['Node4',
    [ 'Node1','Node2','Node3','Node4' ] ,
    [ 1, 1, 1, 0,   1, 1, 1, 0,   1, 1, 1, 0,   1, 1, 1, 0]],
]

test2_bn = [
#                                --- EACH COLULMN REPRESENTS ---
#                                --- STATE OF (x1,x2,x3,x4)  ---
#                                1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
#                                1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0
#                                1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0
#                                1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
  ['x1', ['x1','x2','x3','x4'], [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1]],
  ['x2', ['x1','x2','x3','x4'], [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1]],
  ['x3', ['x1','x2','x3','x4'], [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1]],
  ['x4', ['x1','x2','x3','x4'], [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1]],
 ]
