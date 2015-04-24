import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.append('../')

# Import deft modules
import laplacian

# Make 1d laplacians
alphas = [1,2,3,4,5]
op_types = ['1d_bilateral', '1d_periodic']
Gs = [30,100,300]
h = 1.0
directory = 'laplacians/'

for alpha in alphas:
    for op_type in op_types:
        for G in Gs:
            file_name = '%s_alpha_%d_G_%d.pickle'%(op_type,alpha,G)
            print 'creating operator %s...'%file_name
            op = laplacian.Laplacian(op_type, alpha, G, h)
            op.save(directory + file_name)