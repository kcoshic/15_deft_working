import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Add parent directory to path
import sys
sys.path.append('../code/')
sys.path.append('../sim/')


# Import deft modules
import deft_1d
import simulate_data_1d
import utils

plt.close('all')

# Generate data
N = 3000
G = 300
eG = 100
alpha = 3
data_type = 'wide'

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]


# Simulate data and get default deft settings
data, defaults = simulate_data_1d.run(data_type,N)

# Get bounding box
bbox = [defaults['box_min'], defaults['box_max']]

#
# Do DEFT density estimation
#

results = deft_1d.run(data, G=G, alpha=alpha, \
    bbox=bbox, \
    periodic=False,
    num_samples=100,
    print_t=True)

# Provide feedback
print '\n%s, alpha=%d'%(data_type, alpha)

# Compute true density
xs = results.bin_centers
Q_true = np.zeros(G)
for i, x in enumerate(xs):
    Q_true[i] = eval(defaults['pdf_py'])
Q_true /= results.h*sum(Q_true)

#
# Estimate entropy
#

# Compute the entropy of the sampled densities
Qs = results.Q_samples*results.h
num_samples = Qs.shape[1]
entropies = sp.array([sp.stats.entropy(Qs[:,i]) for i in range(num_samples)])

true_e = sp.stats.entropy(Q_true*results.h)

# Make sure all entropies are finite
assert all(np.isfinite(entropies))

ebox = [0,sp.log(G)]
mean_e = np.mean(entropies)
std_e = np.std(entropies)

# Adjust ebox to data
ebox[0] = max(0, min(entropies)-1*std_e)
ebox[0] = min(ebox[0], true_e)
ebox[1] = min(ebox[1], max(entropies)+1*std_e)
ebox[1] = max(ebox[1], true_e)
eh, es, e_edges = utils.grid_info_from_bbox_and_G(ebox, eG)
eR, _ = utils.histogram_counts_1d(entropies, G=eG, bbox=ebox, normalized=True)

# Now do density estimation for the entropies
eresults = deft_1d.run(entropies, G=eG, bbox=ebox, alpha=3, periodic=False, num_samples=0)

########

# Plot histogram with density estimate on top
plt.figure(figsize=[6, 6])

#
# Show densitye estimates
#
ax = plt.subplot(2,1,1)

# Plot histogram density
left_bin_edges = results.bin_edges[:-1]
plt.bar(left_bin_edges, results.R, \
    width=results.h, linewidth=0, color=gray, zorder=0)

# Plot error bars on density estimate
#plt.fill_between(results.bin_centers,
#    results.Q_ub, results.Q_lb, color=lightblue, 
#    zorder=1, alpha=0.5, linewidth=0)

# Plot deft density estimate
plt.plot(xs, results.Q_samples, \
    color=blue, linewidth=1, alpha=0.2, zorder=2)

# Plot the true density
plt.plot(xs, Q_true, color='k', linewidth=2)

# Tidy up the plot
plt.yticks([])
#plt.xticks([])
plt.ylim([0, 1.2*max(results.Q_star)])
plt.xlim(results.bbox)
t = results.deft_1d_compute_time
plt.title("%s, $\\alpha = %d$, t=%1.2f sec"%(data_type, alpha, t), \
    fontsize=10)

#
# Show entropy estimate
# 
plt.subplot(2,1,2)

left_e_edges = e_edges[:-1]
plt.bar(left_e_edges, eR, width=eh, linewidth=0, color=gray, zorder=0)
plt.xlim(ebox)

plt.plot(eresults.bin_centers, eresults.Q_star, color=orange, linewidth=3)

yl = [0, 1.2*max(eresults.Q_star)]
plt.plot([true_e, true_e], yl, '--k', linewidth=2)
plt.ylim(yl)
plt.yticks([])
plt.title("$%.3f \\pm %.3f$ bits"%(mean_e, std_e))

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.show()
plt.tight_layout() # Needed so plot is drawn tollerably

# Save plot
plt.savefig('report.test_deft_1d.png')