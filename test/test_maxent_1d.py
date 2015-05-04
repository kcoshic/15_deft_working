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
import maxent_1d
import utils

import simulate_data_1d

plt.close('all')

# Generate data
N = 1000
G = 100
alphas = [1, 2, 3, 5, 10, 20, 30, 50]
data_types = ['uniform', 'exponential', 'gaussian', 'wide']

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]

# Plot histogram with density estimate on top
plt.figure(figsize=[ 11.55,  10.25])

num_rows = len(alphas)
num_cols = len(data_types)

for d, data_type in enumerate(data_types):

    data, settings = simulate_data_1d.run(data_type,N)
    box = [settings['box_min'], settings['box_max']]

    # Histogram data
    R, xs = utils.histogram_counts_1d(data, G, bbox=box, normalized=True)
    h = xs[1]-xs[0]

    for a, alpha in enumerate(alphas):

        ax = plt.subplot(num_rows, num_cols, num_cols*a + d + 1)

        # Get basis defining moments to constrain
        basis = utils.legendre_basis_1d(G,alpha)

        # Compute maxent distribution for histogram
        start_time = time.clock()
        results = maxent_1d.run(data, alpha, G=G, bbox=box)
        Q = results.Q_maxent
        end_time = time.clock()
        print '\n%s, alpha=%d'%(data_type, alpha)
        print 'compute_maxent_prob took %f sec'%(end_time-start_time)

        xs_le = utils.left_edges_from_centers(xs)
        xl = utils.bounding_box_from_centers(xs)

        plt.bar(xs_le, R, width=h, linewidth=0, color=gray)
        plt.plot(xs, Q, color=lightblue, linewidth=3)

        plt.yticks([])
        plt.xticks([])
        plt.ylim([0, 1.2*max(Q)])
        plt.xlim(xl)
        plt.title("%s, $\\alpha = %d$"%(data_type, alpha))

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.show()
plt.tight_layout() # Needed so plot is drawn tollerably

# Save plot
plt.savefig('report.test_maxent_1d.png')