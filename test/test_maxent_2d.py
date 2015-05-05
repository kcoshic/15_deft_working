import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory to path
import sys
sys.path.append('../code/')
sys.path.append('../sim/')

# Import deft modules
import laplacian
import maxent
import utils

import simulate_data_2d

plt.close('all')

# Generate data
N = 50
num_bins = [28,28]
alphas = [3,5,7,9,11]
data_types = '''
digit
'''.split()

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]

# Plot histogram with density estimate on top
plt.figure(figsize=[ 11.55,  10.25])

num_rows = len(alphas)
num_cols = 2*len(data_types)
Gx = num_bins[0]
Gy = num_bins[1]
G = Gx*Gy
for d, data_type in enumerate(data_types):

    data, settings = simulate_data_2d.run(data_type,N)
    xbox = [settings['box_xmin'], settings['box_xmax']]
    ybox = [settings['box_ymin'], settings['box_ymax']]
    box = [xbox,ybox]

    # Histogram data
    R_2d, xs, ys = utils.histogram_2d(data, box, num_bins, normalized=True)
    R = R_2d.ravel()
    hx = xs[1]-xs[0]
    hy = ys[1]-ys[0]

    for a, alpha in enumerate(alphas):

        # Get basis defining moments to constrain
        basis = utils.legendre_basis_2d(Gx,Gy,alpha)

        # Compute maxent distribution for histogram
        start_time = time.clock()
        Q, num_corrector_steps, num_backtracks = \
            maxent.compute_maxent_prob_2d(R, basis, grid_spacing=[hx,hy], \
                report_num_steps=True)
        end_time = time.clock()
        print '\n%s, alpha=%d'%(data_type, alpha)
        print 'compute_maxent_prob took %f sec'%(end_time-start_time)
        print 'num_corrector_steps == %d'%num_corrector_steps
        print 'num_backtracks == %d'%num_backtracks

        # Make Q 2d
        Q_2d = Q.reshape(num_bins)

        # Calibrate both R and Q to the same scale
        cl = [0.0, max(Q)]
        cmap = plt.get_cmap('bone')

        # Plot histogram
        ax = plt.subplot(num_rows, num_cols, num_cols*a + 2*d + 1)
        plt.imshow(R_2d.T, interpolation='nearest', cmap=cmap)
        plt.clim(cl)
        plt.title("$\\alpha = %d$"%(alpha))
        plt.yticks([])
        plt.xticks([])

        # Plot density
        ax = plt.subplot(num_rows, num_cols, num_cols*a + 2*d + 2)
        plt.imshow(Q_2d.T, interpolation='nearest', cmap=cmap)
        plt.clim(cl)
        plt.yticks([])
        plt.xticks([])

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.show()
plt.tight_layout() # Needed so plot is drawn tollerably

# Save plot
plt.savefig('report.test_maxent_2d.png')