import shutil
import os
import matplotlib.pyplot as plt

shutil.rmtree('laplacians')
os.mkdir('laplacians')

plt.close('all')

print '\nrunning make_1d_laplacians.py ...'
execfile('make_1d_laplacians.py')

print '\nrunning make_2d_laplacians.py ...'
execfile('make_2d_laplacians.py')

print '\nrunning test_1d_laplacians.py ...'
execfile('test_1d_laplacians.py')

print '\nrunning test_2d_laplacians.py ...'
execfile('test_2d_laplacians.py')