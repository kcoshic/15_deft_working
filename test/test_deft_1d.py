import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory to path
import sys
sys.path.append('../')

# Import deft modules
import deft_1d
import simulate_data_1d

plt.close('all')

# Generate data
N = 100
G = 100
#alphas = [3]
#data_types = ['wide']
alphas = [1,2,3]

data_types = '''
uniform 
exponential
gaussian
wide
gamma
vonmises
'''.split()

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]

# Plot histogram with density estimate on top
plt.figure(figsize=[12.5, 8])

#alphas = [1,2,3]
#data_types = data_types[3:5]

num_cols = len(alphas)
num_rows = len(data_types)

# Iterate over types of distributions
for d, data_type in enumerate(data_types):

    # Simulate data and get default deft settings
    data, defaults = simulate_data_1d.run(data_type,N)

    # Iterate over value sfor alpha
    for a, alpha in enumerate(alphas):

        # Specify subplot
        ax = plt.subplot(num_rows, num_cols, num_cols*d + a + 1)

        bbox = [defaults['box_min'], defaults['box_max']]

        # Do density estimation
        results = deft_1d.run(data, G=G, alpha=alpha, \
            bbox=bbox, \
            periodic=defaults['periodic'])

        # Plot histogram density
        left_bin_edges = results.bin_edges[:-1]
        plt.bar(left_bin_edges, results.R, \
            width=results.h, linewidth=0, color=gray, zorder=0)

        # Plot error bars on density estimate
        plt.fill_between(results.bin_centers,
            results.Q_ub, results.Q_lb, color=lightblue, 
            zorder=1, alpha=0.5, linewidth=0)

        # Plot deft density estimate
        plt.plot(results.bin_centers, results.Q_star, \
            color=blue, linewidth=3, zorder=2)

        # Plot error bars on density estimate
        #plt.plot(results.bin_centers, results.Q_lb, \
        #   color=lightblue, linewidth=1)
        #plt.plot(results.bin_centers, results.Q_ub, \
        #   color=lightblue, linewidth=1)

        # Tidy up the plot
        plt.yticks([])
        plt.xticks([])
        plt.ylim([0, 1.2*max(results.Q_star)])
        plt.xlim(results.bbox)
        t = results.execution_time
        plt.title("%s, $\\alpha = %d$, t=%1.2f sec"%(data_type, alpha, t), \
            fontsize=10)

        # Provide feedback
        print '\n%s, alpha=%d'%(data_type, alpha)
        print 'compute_maxent_prob took %f sec'%t

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.show()
plt.tight_layout() # Needed so plot is drawn tollerably

# Save plot
plt.savefig('report.test_deft_1d.png')