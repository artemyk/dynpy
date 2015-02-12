"""This module provides several sample networks and dynamical systems for use
in testing and development.

Right now it includes:

Several published boolean networks (see source)

``karateclub_net``: A 34-node graph representing Zachary's karate-club network.
It is in numpy array format.

``test_bn``: A simple 4-node BN for testing

``test2_bn``: Another simple 4-node BN for testing

"""

import numpy as np



# Adapted from http://people.kth.se/~dubrova/bns.html (cnet files provided for BNS)
#  # Boolean network model of the control of flower morphogenesis in Arabidobis thaliana
#  # "From Genes to Flower Patterns and Evolution: Dynamic Models of Gene Regulatory Networks"
#  # A. Chaos, M. Aldana, C. Espinosa-Soto, B. G. P. de Leon, A. G. Arroyo, E. R. Alvarez-Buylla,
#  # Journal of Plant Growth Regulation, vol. 25, n. 4, 2006, pp. 278-289
#  # The functions associated to nodes are minimized
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
 ['AG', ['AP1', 'LFY', 'AP2', 'WUS', 'AG', 'TFL1', 'SEP'], lambda AP1, LFY, AP2, WUS, AG, TFL1, SEP: ((LFY and AG and SEP) or (not AP2 and not TFL1) or (LFY and not True) or (LFY and not True) or (LFY and WUS) or (not AP1 and LFY) or (LFY and not AP2))],
 ['TFL1', ['AP1', 'EMF1', 'LFY', 'AP2'], lambda AP1, EMF1, LFY, AP2: ((not AP1 and EMF1 and not LFY))],
 ['PI', ['AP1', 'LFY', 'AG', 'PI', 'SEP', 'AP3'], lambda AP1, LFY, AG, PI, SEP, AP3: ((AP1 and PI and SEP and AP3) or (AG and PI and SEP and AP3) or (LFY and AG) or (LFY and AP3))],
 ['SEP', ['LFY'], lambda LFY: ((LFY))]
]



# Adapted from http://people.kth.se/~dubrova/bns.html (cnet files provided for BNS)
#  # Boolean network model of the control of the budding yeast cell cycle regulation from
#  # "The yeast cell-cycle network is robustly designed",
#  # Fangting Li, Tao Long, Ying Lu, Qi Ouyang, Chao Tang,
#  #  PNAS  April 6, 2004,  vol. 101  no. 14  4781-4786.
budding_yeast_bn = [
 ['Cln3', [], lambda : ((False))],
 ['MBF', ['Cln3', 'MBF', 'Clb1,2'], lambda Cln3, MBF, Clb1_2: ((MBF and not Clb1_2) or (Cln3 and MBF) or (Cln3 and not Clb1_2))],
 ['SBF', ['Cln3', 'SBF', 'Clb1,2'], lambda Cln3, SBF, Clb1_2: ((SBF and not Clb1_2) or (Cln3 and SBF) or (Cln3 and not Clb1_2))],
 ['Cln1,2', ['SBF'], lambda SBF: ((SBF))],
 ['Sic1', ['Cln1,2', 'Sic1', 'Clb5,6', 'Clb1,2', 'Cdc20', 'Swi5'], lambda Cln1_2, Sic1, Clb5_6, Clb1_2, Cdc20, Swi5: ((not Cln1_2 and Sic1 and not Clb5_6 and not Clb1_2) or (Sic1 and not Clb5_6 and not Clb1_2 and Cdc20) or (not Cln1_2 and Sic1 and not Clb1_2 and Cdc20) or (not Cln1_2 and Sic1 and not Clb5_6 and Cdc20) or (Sic1 and not Clb5_6 and not Clb1_2 and Swi5) or (not Cln1_2 and Sic1 and not Clb1_2 and Swi5) or (not Cln1_2 and Sic1 and not Clb5_6 and Swi5) or (Sic1 and not Clb1_2 and Cdc20 and Swi5) or (Sic1 and not Clb5_6 and Cdc20 and Swi5) or (not Cln1_2 and Sic1 and Cdc20 and Swi5) or (not Cln1_2 and not Clb5_6 and not Clb1_2 and Swi5) or (not Clb5_6 and not Clb1_2 and Cdc20 and Swi5) or (not Cln1_2 and not Clb1_2 and Cdc20 and Swi5) or (not Cln1_2 and not Clb5_6 and Cdc20 and Swi5) or (not Cln1_2 and not Clb5_6 and not Clb1_2 and Cdc20))],
 ['Swi5', ['Clb1,2', 'Mcm1', 'Cdc20', 'Swi5'], lambda Clb1_2, Mcm1, Cdc20, Swi5: ((not Clb1_2 and Cdc20) or (Mcm1 and Cdc20) or (not Clb1_2 and Mcm1))],
 ['Cdc20', ['Clb1,2', 'Mcm1'], lambda Clb1_2, Mcm1: ((Mcm1) or (Clb1_2))],
 ['Clb5,6', ['MBF', 'Sic1', 'Clb5,6', 'Cdc20'], lambda MBF, Sic1, Clb5_6, Cdc20: ((not Sic1 and Clb5_6 and not Cdc20) or (MBF and Clb5_6 and not Cdc20) or (MBF and not Sic1 and Clb5_6) or (MBF and not Sic1 and not Cdc20))],
 ['Cdh1', ['Cln1,2', 'Clb5,6', 'Cdh1', 'Clb1,2', 'Cdc20'], lambda Cln1_2, Clb5_6, Cdh1, Clb1_2, Cdc20: ((not Cln1_2 and not Clb5_6 and Cdh1 and not Clb1_2) or (not Clb5_6 and Cdh1 and not Clb1_2 and Cdc20) or (not Cln1_2 and Cdh1 and not Clb1_2 and Cdc20) or (not Cln1_2 and not Clb5_6 and Cdh1 and Cdc20) or (not Cln1_2 and not Clb5_6 and not Clb1_2 and Cdc20))],
 ['Clb1,2', ['Sic1', 'Clb5,6', 'Cdh1', 'Clb1,2', 'Mcm1', 'Cdc20'], lambda Sic1, Clb5_6, Cdh1, Clb1_2, Mcm1, Cdc20: ((not Sic1 and Clb5_6 and not Cdh1 and Clb1_2) or (not Sic1 and not Cdh1 and Clb1_2 and Mcm1) or (Clb5_6 and not Cdh1 and Clb1_2 and Mcm1) or (not Sic1 and Clb5_6 and Clb1_2 and Mcm1) or (not Sic1 and not Cdh1 and Clb1_2 and not Cdc20) or (Clb5_6 and not Cdh1 and Clb1_2 and not Cdc20) or (not Sic1 and Clb5_6 and Clb1_2 and not Cdc20) or (not Cdh1 and Clb1_2 and Mcm1 and not Cdc20) or (not Sic1 and Clb1_2 and Mcm1 and not Cdc20) or (Clb5_6 and Clb1_2 and Mcm1 and not Cdc20) or (not Sic1 and Clb5_6 and not Cdh1 and not Cdc20) or (not Sic1 and not Cdh1 and Mcm1 and not Cdc20) or (Clb5_6 and not Cdh1 and Mcm1 and not Cdc20) or (not Sic1 and Clb5_6 and Mcm1 and not Cdc20) or (not Sic1 and Clb5_6 and not Cdh1 and Mcm1))],
 ['Mcm1', ['Clb5,6', 'Clb1,2'], lambda Clb5_6, Clb1_2: ((Clb1_2) or (Clb5_6))],
]



# Adapted from http://people.kth.se/~dubrova/bns.html (cnet files provided for BNS)
#  # Boolean network model of Drosophila melanogaster from
#  # "The topology of the regulatory interactions predicts the expression pattern of the
#  # segment polarity genes in Drosophila melanogaster", R. Albert and H. G. Othmer,
#  # Journal of Theoretical Biology, 2003, vol. 223, no. 1, pp. 1-18.
#  # The number of cells is reducecd from 12 to 4
# Modified to so that cell 4 now connects to cell 1 (i.e. ring topology)

drosophila4_bn = [
 ['wg1', ['wg1', 'CIA1', 'CIR1'], lambda wg1, CIA1, CIR1: ((CIA1 and False and not CIR1) or (wg1 and (CIA1 or False) and not CIR1))],
 ['wg2', ['wg2', 'CIA2', 'CIR2'], lambda wg2, CIA2, CIR2: ((CIA2 and False and not CIR2) or (wg2 and (CIA2 or False) and not CIR2))],
 ['wg3', ['wg3', 'CIA3', 'CIR3'], lambda wg3, CIA3, CIR3: ((CIA3 and True and not CIR3) or (wg3 and (CIA3 or True) and not CIR3))],
 ['wg4', ['wg4', 'CIA4', 'CIR4'], lambda wg4, CIA4, CIR4: ((CIA4 and True and not CIR4) or (wg4 and (CIA4 or True) and not CIR4))],
 ['WG1', ['wg1'], lambda wg1: ((wg1))],
 ['WG2', ['wg2'], lambda wg2: ((wg2))],
 ['WG3', ['wg3'], lambda wg3: ((wg3))],
 ['WG4', ['wg4'], lambda wg4: ((wg4))],
 ['en1', ['WG4', 'WG2'], lambda WG4, WG2: ((WG2 and not False) or (WG4 and not False))],
 ['en2', ['WG1', 'WG3'], lambda WG1, WG3: ((WG3 and not False) or (WG1 and not False))],
 ['en3', ['WG2', 'WG4'], lambda WG2, WG4: ((WG4 and not True) or (WG2 and not True))],
 ['en4', ['WG3', 'WG1'], lambda WG3, WG1: ((WG3 and not True) or (WG1 and not True))],
 ['EN1', ['en1'], lambda en1: ((en1))],
 ['EN2', ['en2'], lambda en2: ((en2))],
 ['EN3', ['en3'], lambda en3: ((en3))],
 ['EN4', ['en4'], lambda en4: ((en4))],
 ['hh1', ['EN1', 'CIR1'], lambda EN1, CIR1: ((EN1 and not CIR1))],
 ['hh2', ['EN2', 'CIR2'], lambda EN2, CIR2: ((EN2 and not CIR2))],
 ['hh3', ['EN3', 'CIR3'], lambda EN3, CIR3: ((EN3 and not CIR3))],
 ['hh4', ['EN4', 'CIR4'], lambda EN4, CIR4: ((EN4 and not CIR4))],
 ['HH1', ['hh1'], lambda hh1: ((hh1))],
 ['HH2', ['hh2'], lambda hh2: ((hh2))],
 ['HH3', ['hh3'], lambda hh3: ((hh3))],
 ['HH4', ['hh4'], lambda hh4: ((hh4))],
 ['ptc1', ['CIA1', 'EN1', 'CIR1'], lambda CIA1, EN1, CIR1: ((CIA1 and not EN1 and not CIR1))],
 ['ptc2', ['CIA2', 'EN2', 'CIR2'], lambda CIA2, EN2, CIR2: ((CIA2 and not EN2 and not CIR2))],
 ['ptc3', ['CIA3', 'EN3', 'CIR3'], lambda CIA3, EN3, CIR3: ((CIA3 and not EN3 and not CIR3))],
 ['ptc4', ['CIA4', 'EN4', 'CIR4'], lambda CIA4, EN4, CIR4: ((CIA4 and not EN4 and not CIR4))],
 ['PTC1', ['ptc1', 'PTC1', 'HH4', 'HH2'], lambda ptc1, PTC1, HH4, HH2: ((PTC1 and not HH4 and not HH2) or (ptc1))],
 ['PTC2', ['ptc2', 'PTC2', 'HH1', 'HH3'], lambda ptc2, PTC2, HH1, HH3: ((PTC2 and not HH1 and not HH3) or (ptc2))],
 ['PTC3', ['ptc3', 'PTC3', 'HH2', 'HH4'], lambda ptc3, PTC3, HH2, HH4: ((PTC3 and not HH2 and not HH4) or (ptc3))],
 ['PTC4', ['ptc4', 'PTC4', 'HH3', 'HH1'], lambda ptc4, PTC4, HH3, HH1: ((PTC4 and not HH1 and not HH3) or (ptc4))],
 ['ci1', ['EN1'], lambda EN1: ((not EN1))],
 ['ci2', ['EN2'], lambda EN2: ((not EN2))],
 ['ci3', ['EN3'], lambda EN3: ((not EN3))],
 ['ci4', ['EN4'], lambda EN4: ((not EN4))],
 ['CI1', ['ci1'], lambda ci1: ((ci1))],
 ['CI2', ['ci2'], lambda ci2: ((ci2))],
 ['CI3', ['ci3'], lambda ci3: ((ci3))],
 ['CI4', ['ci4'], lambda ci4: ((ci4))],
 ['CIA1', ['CI1', 'PTC1', 'HH4', 'hh4', 'HH2', 'hh2'], lambda CI1, PTC1, HH4, hh4, HH2, hh2: ((CI1 and hh4) or (CI1 and HH4) or (CI1 and hh2) or (CI1 and HH2) or (CI1 and not PTC1))],
 ['CIA2', ['CI2', 'PTC2', 'HH1', 'HH3', 'hh1', 'hh3'], lambda CI2, PTC2, HH1, HH3, hh1, hh3: ((CI2 and hh3) or (CI2 and hh1) or (CI2 and HH3) or (CI2 and HH1) or (CI2 and not PTC2))],
 ['CIA3', ['CI3', 'PTC3', 'HH2', 'HH4', 'hh2', 'hh4'], lambda CI3, PTC3, HH2, HH4, hh2, hh4: ((CI3 and hh4) or (CI3 and hh2) or (CI3 and HH4) or (CI3 and HH2) or (CI3 and not PTC3))],
 ['CIA4', ['CI4', 'PTC4', 'HH3', 'hh3', 'HH1', 'hh1'], lambda CI4, PTC4, HH3, hh3, HH1, hh1: ((CI4 and hh1) or (CI4 and HH1) or (CI4 and hh3) or (CI4 and HH3) or (CI4 and not PTC4))],
 ['CIR1', ['CI1', 'PTC1', 'HH4', 'hh4', 'HH2', 'hh2'], lambda CI1, PTC1, HH4, hh4, HH2, hh2: ((CI1 and PTC1 and not HH2 and not hh2 and not HH4 and not hh4))],
 ['CIR2', ['CI2', 'PTC2', 'HH1', 'HH3', 'hh1', 'hh3'], lambda CI2, PTC2, HH1, HH3, hh1, hh3: ((CI2 and PTC2 and not HH1 and not HH3 and not hh1 and not hh3))],
 ['CIR3', ['CI3', 'PTC3', 'HH2', 'HH4', 'hh2', 'hh4'], lambda CI3, PTC3, HH2, HH4, hh2, hh4: ((CI3 and PTC3 and not HH2 and not HH4 and not hh2 and not hh4))],
 ['CIR4', ['CI4', 'PTC4', 'HH3', 'hh3', 'HH1', 'hh1'], lambda CI4, PTC4, HH3, hh3, HH1, hh1: ((CI4 and PTC4 and not HH3 and not hh3 and not HH1 and not hh1))]
]



# Adapted from http://people.kth.se/~dubrova/bns.html (cnet files provided for BNS)
#  # Boolean network model of the control of the fission yeast cell cycle regulation from
#  # "Boolean Network Model Predicts Cell Cycle Sequence of Fission Yeast",
#  # M. I. Davidich, S. Bornholdt, PLoS ONE. 2008 Feb 27, 3(2):e1672.
fission_yeast_bn = [
 ['SK', [], lambda : ((False))],
 ['Ste9', ['SK', 'Ste9', 'Cdc2_Cdc13', 'PP', 'Cdc2_Cdc13_'], lambda SK, Ste9, Cdc2_Cdc13, PP, Cdc2_Cdc13_: ((not SK and Ste9 and not Cdc2_Cdc13 and not Cdc2_Cdc13_) or (Ste9 and not Cdc2_Cdc13 and PP and not Cdc2_Cdc13_) or (not SK and Ste9 and PP and not Cdc2_Cdc13_) or (not SK and Ste9 and not Cdc2_Cdc13 and PP) or (not SK and not Cdc2_Cdc13 and PP and not Cdc2_Cdc13_))],
 ['Cdc2_Cdc13', ['Ste9', 'Cdc2_Cdc13', 'Rum1', 'Slp1'], lambda Ste9, Cdc2_Cdc13, Rum1, Slp1: ((Cdc2_Cdc13 and not Rum1 and not Slp1) or (not Ste9 and Cdc2_Cdc13 and not Slp1) or (not Ste9 and Cdc2_Cdc13 and not Rum1) or (not Ste9 and not Rum1 and not Slp1))],
 ['Rum1', ['SK', 'Cdc2_Cdc13', 'Rum1', 'PP', 'Cdc2_Cdc13_'], lambda SK, Cdc2_Cdc13, Rum1, PP, Cdc2_Cdc13_: ((not Cdc2_Cdc13 and Rum1 and PP and Cdc2_Cdc13_) or (not SK and not Cdc2_Cdc13 and Rum1 and not Cdc2_Cdc13_) or (not SK and Rum1 and PP and not Cdc2_Cdc13_) or (not SK and not Cdc2_Cdc13 and PP and not Cdc2_Cdc13_))],
 ['PP', ['Slp1'], lambda Slp1: ((Slp1))],
 ['Cdc25', ['Cdc2_Cdc13', 'PP', 'Cdc25'], lambda Cdc2_Cdc13, PP, Cdc25: ((not PP and Cdc25) or (Cdc2_Cdc13 and Cdc25) or (Cdc2_Cdc13 and not PP))],
 ['Slp1', ['Cdc2_Cdc13_'], lambda Cdc2_Cdc13_: ((Cdc2_Cdc13_))],
 ['Wee1_Mik1', ['Cdc2_Cdc13', 'PP', 'Wee1_Mik1'], lambda Cdc2_Cdc13, PP, Wee1_Mik1: ((not Cdc2_Cdc13 and Wee1_Mik1) or (PP and Wee1_Mik1) or (not Cdc2_Cdc13 and PP))],
 ['Cdc2_Cdc13_', ['Ste9', 'Rum1', 'Cdc25', 'Slp1', 'Wee1_Mik1', 'Cdc2_Cdc13_'], lambda Ste9, Rum1, Cdc25, Slp1, Wee1_Mik1, Cdc2_Cdc13_: ((not Ste9 and not Rum1 and Cdc25 and not Slp1 and not Wee1_Mik1 and Cdc2_Cdc13_))]
]



# Adapted from http://people.kth.se/~dubrova/bns.html (cnet files provided for BNS)
#  # Boolean network model of the control of the mammalian cell cycle from
#  # "Dynamical Analysis of a Generic Boolean Model for the Control of the
#  # Mammalian Cell Cycle", A. Faure, A. Naldi, C. Chaouiya, D. Thieffry,
#  # Bioinformatics, 2006, vol. 22, no. 14, pp. e124-e131.
mammalian_bn = [
 ['CycD', ['CycD'], lambda CycD: ((CycD))],
 ['CycE', ['Rb', 'E2F'], lambda Rb, E2F: ((not Rb and E2F))],
 ['Rb', ['CycD', 'CycE', 'CycA', 'p27', 'CycB'], lambda CycD, CycE, CycA, p27, CycB: ((not CycD and not CycE and not CycA and not CycB) or (not CycD and p27 and not CycB))],
 ['E2F', ['Rb', 'CycA', 'p27', 'CycB'], lambda Rb, CycA, p27, CycB: ((not Rb and p27 and not CycB) or (not Rb and not CycA and not CycB))],
 ['CycA', ['Rb', 'E2F', 'CycA', 'Cdc20', 'UbcH10', 'Cdh1'], lambda Rb, E2F, CycA, Cdc20, UbcH10, Cdh1: ((not Rb and CycA and not Cdc20 and not Cdh1) or (not Rb and E2F and not Cdc20 and not Cdh1) or (not Rb and CycA and not Cdc20 and not UbcH10) or (not Rb and E2F and not Cdc20 and not UbcH10))],
 ['p27', ['CycD', 'CycE', 'CycA', 'p27', 'CycB'], lambda CycD, CycE, CycA, p27, CycB: ((not CycD and not CycA and p27 and not CycB) or (not CycD and not CycE and p27 and not CycB) or (not CycD and not CycE and not CycA and not CycB))],
 ['Cdc20', ['CycB'], lambda CycB: ((CycB))],
 ['UbcH10', ['CycA', 'Cdc20', 'UbcH10', 'Cdh1', 'CycB'], lambda CycA, Cdc20, UbcH10, Cdh1, CycB: ((UbcH10 and CycB) or (Cdc20 and UbcH10) or (CycA and UbcH10) or (not Cdh1))],
 ['Cdh1', ['CycA', 'p27', 'Cdc20', 'CycB'], lambda CycA, p27, Cdc20, CycB: ((p27 and not CycB) or (not CycA and not CycB) or (Cdc20))],
 ['CycB', ['Cdc20', 'Cdh1'], lambda Cdc20, Cdh1: ((not Cdc20 and not Cdh1))]
]



# Adapted from http://people.kth.se/~dubrova/bns.html (cnet files provided for BNS)
#  # Boolean network model of the T-cell receptor signalling pathway from
#  # "A methodology for the structural and functional analysis of signaling and
#  # regulatory networks", S. Klamt, J. Saez-Rodriguez, J. A. Lindquist, L. Simeoni, E. D. Gilles,
#  # JBMC Bioinformatics 7: 56, 2006.
tcr_bn = [
 ['CD45', ['CD45'], lambda CD45: ((CD45))],
 ['CD8', ['CD8'], lambda CD8: ((CD8))],
 ['TCRlig', ['TCRlig'], lambda TCRlig: ((TCRlig))],
 ['TCRbind', ['TCRlig', 'cCbl'], lambda TCRlig, cCbl: ((TCRlig and not cCbl))],
 ['PAGCsk', ['Fyn', 'TCRbind'], lambda Fyn, TCRbind: ((not TCRbind) or (Fyn))],
 ['LCK', ['CD45', 'CD8', 'PAGCsk'], lambda CD45, CD8, PAGCsk: ((CD45 and CD8 and not PAGCsk))],
 ['Fyn', ['CD45', 'TCRbind', 'LCK'], lambda CD45, TCRbind, LCK: ((CD45 and LCK) or (CD45 and TCRbind))],
 ['Rlk', ['LCK'], lambda LCK: ((LCK))],
 ['TCRphos', ['TCRbind', 'LCK', 'Fyn'], lambda TCRbind, LCK, Fyn: ((TCRbind and LCK) or (Fyn))],
 ['ZAP70', ['LCK', 'TCRphos', 'cCbl'], lambda LCK, TCRphos, cCbl: ((LCK and TCRphos and not cCbl))],
 ['cCbl', ['ZAP70'], lambda ZAP70: ((ZAP70))],
 ['Itk', ['ZAP70', 'Slp76'], lambda ZAP70, Slp76: ((ZAP70 and Slp76))],
 ['LAT', ['ZAP70'], lambda ZAP70: ((ZAP70))],
 ['Gads', ['LAT'], lambda LAT: ((LAT))],
 ['Slp76', ['Gads'], lambda Gads: ((Gads))],
 ['PLCg_b', ['LAT'], lambda LAT: ((LAT))],
 ['Grb2Sos', ['LAT'], lambda LAT: ((LAT))],
 ['DAG', ['PLCg_a'], lambda PLCg_a: ((PLCg_a))],
 ['PLCg_a', ['Rlk', 'ZAP70', 'Itk', 'Slp76', 'PLCg_b'], lambda Rlk, ZAP70, Itk, Slp76, PLCg_b: ((ZAP70 and Itk and Slp76 and PLCg_b) or (Rlk and ZAP70 and Slp76 and PLCg_b))],
 ['Ras', ['Grb2Sos', 'RasGRP1'], lambda Grb2Sos, RasGRP1: ((RasGRP1) or (Grb2Sos))],
 ['RasGRP1', ['DAG', 'PKCth'], lambda DAG, PKCth: ((DAG and PKCth))],
 ['PKCth', ['DAG'], lambda DAG: ((DAG))],
 ['IP3', ['PLCg_a'], lambda PLCg_a: ((PLCg_a))],
 ['Raf', ['Ras'], lambda Ras: ((Ras))],
 ['MEK', ['Raf'], lambda Raf: ((Raf))],
 ['Ca', ['IP3'], lambda IP3: ((IP3))],
 ['ERK', ['MEK'], lambda MEK: ((MEK))],
 ['SEK', ['PKCth'], lambda PKCth: ((PKCth))],
 ['IKK', ['PKCth'], lambda PKCth: ((PKCth))],
 ['Calcin', ['Ca'], lambda Ca: ((Ca))],
 ['Rsk', ['ERK'], lambda ERK: ((ERK))],
 ['Fos', ['ERK'], lambda ERK: ((ERK))],
 ['JNK', ['SEK'], lambda SEK: ((SEK))],
 ['Ikb', ['IKK'], lambda IKK: ((not IKK))],
 ['CREB', ['Rsk'], lambda Rsk: ((Rsk))],
 ['Jun', ['JNK'], lambda JNK: ((JNK))],
 ['CRE', ['CREB'], lambda CREB: ((CREB))],
 ['AP1', ['Fos', 'Jun'], lambda Fos, Jun: ((Fos and Jun))],
 ['NFkB', ['Ikb'], lambda Ikb: ((not Ikb))],
 ['NFAT', ['Calcin'], lambda Calcin: ((Calcin))]
]



# Adapted from http://people.kth.se/~dubrova/bns.html (cnet files provided for BNS)
#  # Boolean network model of the control of T-helper cell differentiation from
#  # "A method for the generation of standardized qualitative dynamical systems
#  # of regulatory networks", L. Mendoza and I. Xenarios
#  # J. Theor. Biol. and Medical Modelling, 2006, vol. 3, no. 13
thelper_bn = [
 ['NFAT', [], lambda : ((False))],
 ['IFN__beta_R', [], lambda : ((False))],
 ['IL_18R', ['STAT6'], lambda STAT6: ((False and not STAT6))],
 ['IRAK', ['IL_18R'], lambda IL_18R: ((IL_18R))],
 ['SOCS1', ['T_bet', 'STAT1'], lambda T_bet, STAT1: ((STAT1) or (T_bet))],
 ['IL_12R', ['STAT6'], lambda STAT6: ((False and not STAT6))],
 ['STAT4', ['IL_12R', 'GATA3'], lambda IL_12R, GATA3: ((IL_12R and not GATA3))],
 ['T_bet', ['T_bet', 'STAT1', 'GATA3'], lambda T_bet, STAT1, GATA3: ((STAT1 and not GATA3) or (T_bet and not GATA3))],
 ['IFN__gamma', ['NFAT', 'IRAK', 'STAT4', 'T_bet', 'STAT3'], lambda NFAT, IRAK, STAT4, T_bet, STAT3: ((T_bet and not STAT3) or (STAT4 and not STAT3) or (IRAK and not STAT3) or (NFAT and not STAT3))],
 ['IFN__gamma_R', ['IFN__gamma'], lambda IFN__gamma: ((IFN__gamma))],
 ['JAK1', ['IFN__gamma_R', 'SOCS1'], lambda IFN__gamma_R, SOCS1: ((IFN__gamma_R and not SOCS1))],
 ['STAT1', ['IFN__beta_R', 'JAK1'], lambda IFN__beta_R, JAK1: ((JAK1) or (IFN__beta_R))],
 ['IL_4', ['GATA3', 'STAT1'], lambda GATA3, STAT1: ((GATA3 and not STAT1))],
 ['IL_4R', ['IL_4', 'SOCS1'], lambda IL_4, SOCS1: ((IL_4 and not SOCS1))],
 ['STAT6', ['IL_4R'], lambda IL_4R: ((IL_4R))],
 ['GATA3', ['STAT6', 'GATA3', 'T_bet'], lambda STAT6, GATA3, T_bet: ((GATA3 and not T_bet) or (STAT6 and not T_bet))],
 ['IL_10', ['GATA3'], lambda GATA3: ((GATA3))],
 ['IL_10R', ['IL_10'], lambda IL_10: ((IL_10))],
 ['STAT3', ['IL_10R'], lambda IL_10R: ((IL_10R))]
]

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

test3_bn = [ 
  ['x1', ['x1','x2','x3','x4'], lambda x1,x2,x3,x4: (not x1 and x2) or (x1 and not x2) or (x1 and x2) or (x3 and x4)],
  ['x2', ['x1','x2','x3','x4'], lambda x1,x2,x3,x4: (not x1 and not x2) or (x1 and not x2) or (x1 and x2) or (x3 and x4)],
  ['x3', ['x1','x2','x3','x4'], lambda x1,x2,x3,x4: (not x3 and x4) or (x3 and not x4) or (x1 and x2) or (x3 and x4)],
  ['x4', ['x1','x2','x3','x4'], lambda x1,x2,x3,x4: (not x3 and not x4) or (x3 and not x4) or (x1 and x2) or (x3 and x4)],
]