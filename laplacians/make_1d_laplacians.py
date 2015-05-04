#!/usr/local/bin/python
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.append('../code/')
sys.path.append('../sim/')

# Import deft modules
import laplacian

# Make 1d laplacians
alphas = [1,2,3]
op_types = ['1d_bilateral', '1d_periodic']
Gs = [20,50,100,200,500,1000,2000]
h = 1.0
directory = '../laplacians/'

for alpha in alphas:
    for op_type in op_types:
        for G in Gs:
            file_name = '%s_alpha_%d_G_%d.pickle'%(op_type,alpha,G)
            print 'creating operator %s...'%file_name
            op = laplacian.Laplacian(op_type, alpha, G, h)
            op.save(directory + file_name)