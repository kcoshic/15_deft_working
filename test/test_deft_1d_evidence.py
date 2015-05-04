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
N = 50
G = 500
eG = 100
alpha = 3
data_type = 'wide'

resolution=3.14E-2
tollerance=1E-3
ell_guess=2.0

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
    errorbars=False,
    print_t=False,
    ell_guess=ell_guess,
    resolution=resolution,
    tollerance=tollerance)

# Provide feedback
print '\n%s, alpha=%d'%(data_type, alpha)
print 'laplacian_compute_time  == %.2f sec'%results.laplacian_compute_time
print 'map_curve_compute_time  == %.2f sec'%results.map_curve_compute_time 
#print 'errorbar_compute_time  == %.2f sec'%results.errorbar_compute_time 
print 'posterior_sample_compute_time  == %.2f sec'%results.\
    posterior_sample_compute_time
print 'copy_compute_time  == %.2f sec'%results.copy_compute_time
print 'entropy_compute_time  == %.2f sec'%results.entropy_compute_time
print 'deft_1d_compute_time == %.2f sec'%results.deft_1d_compute_time

# Compute true density
xs = results.bin_centers
Q_true = np.zeros(G)
for i, x in enumerate(xs):
    Q_true[i] = eval(defaults['pdf_py'])
Q_true /= results.h*sum(Q_true)

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

# Plot posterior samples
#plt.plot(xs, results.Q_samples, \
#    color=blue, linewidth=1, alpha=0.2, zorder=2)

# Plot deft estimate
plt.plot(xs, results.Q_star, \
    color=blue, linewidth=3, zorder=1)

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
# Show evidence ratio
# 
plt.subplot(2,1,2)

# Plot evidence ratio
map_curve = results.map_curve
log_Es = [p.log_E for p in map_curve.points]
ts = [p.t for p in map_curve.points]
t_star = ts[map_curve.i_star]
max_log_E = max(log_Es)
plt.plot(ts[1:-1], log_Es[1:-1], '-o')
plt.ylim([max_log_E-20, max_log_E+2])
plt.title("%d ts, t_star=%.1f, max(log_E)=%.1f"%(len(ts), t_star, max_log_E))

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.show()
plt.tight_layout() # Needed so plot is drawn tollerably

# Save plot
plt.savefig('report.test_deft_1d.png')