import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.append('../')

# Import deft modules
import laplacian

# Make 1d laplacians
alphas = [1,2,3]
op_types = ['2d_bilateral', '2d_periodic']
Gs_per_side = [10,20]
h = 1.0
directory = 'laplacians/'

for alpha in alphas:
    for op_type in op_types:
        for G_per_side in Gs_per_side:
            Gx = G_per_side
            Gy = G_per_side
            file_name = '%s_alpha_%d_G_%dx%d.pickle'%(op_type,alpha,Gx,Gy)
            print 'creating operator %s...'%file_name
            op = laplacian.Laplacian(op_type, alpha, [Gx,Gy], [h,h])
            op.save(directory + file_name)