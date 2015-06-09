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
import deft_core
import utils
import deft_1d

import simulate_data_2d


# Compute mutual information in bits
def mutual_information(Q_xy):
    assert all(Q_xy.ravel() >= 0)
    assert len(Q_xy.shape) == 2
    Gx = Q_xy.shape[0]
    Gy = Q_xy.shape[1]
    Q_xy /= Q_xy.sum()
    Q = Q_xy.ravel()
    Q_x = Q_xy.sum(1)
    Q_y = Q_xy.sum(0)
    H = -sp.sum(Q*sp.log2(Q + utils.TINY_FLOAT64))
    H_x = -sp.sum(Q_x*sp.log2(Q_x + utils.TINY_FLOAT64))
    H_y = -sp.sum(Q_y*sp.log2(Q_y + utils.TINY_FLOAT64))
    I = H_x + H_y - H
    return I


plt.close('all')

# Generate data
N = 100
num_bins = [25,25]
alpha = 4
data_type = 'face' 

resolution = 1E-1
tollerance = 1E-3

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]

# Plot histogram with density estimate on top
plt.figure(figsize=[ 15, 2.5])

num_rows = 1
num_cols = 5
row = 0
Gx = num_bins[0]
Gy = num_bins[1]
G = Gx*Gy

data, settings = simulate_data_2d.run(data_type,N)
xbox = [settings['box_xmin'], settings['box_xmax']]
ybox = [settings['box_ymin'], settings['box_ymax']]
box = [xbox,ybox]

# Retrieve the true distribution
hx, x_centers, x_edges = utils.grid_info_from_bbox_and_G(xbox, Gx)
hy, y_centers, y_edges = utils.grid_info_from_bbox_and_G(ybox, Gy)
ps = sp.zeros([Gx,Gy])
for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        ps[i,j] = eval(settings['pdf_py'])
Q_true = ps/ps.sum()

# Compute the true mutual information
mi_true = mutual_information(Q_true)

# Histogram data
counts_2d, xs, ys = utils.histogram_2d(data, box, num_bins, \
    normalized=False)
counts = counts_2d.ravel()

# Get laplacian to use
Delta = laplacian.Laplacian('2d_bilateral',alpha,num_bins,[1.,1.])

# Compute maxent distribution for histogram
print '\ndata_type=%s, alpha=%d, N=%d'%(data_type, alpha,N)
start_time = time.clock()
results = \
    deft_core.run(counts, Delta, resolution=resolution, 
        tollerance=tollerance, details=True, num_samples=100,
        print_t = True, errorbars=False)
end_time = time.clock()
print 'compute_maxent_prob took %f sec'%(end_time-start_time)

# Compute mutaul information estimate in bits
entropy_start_time = time.clock()
num_samples = results.num_samples
mis = sp.zeros(num_samples)
if num_samples > 1:
    entropies = np.zeros(num_samples)
    for i in range(results.Q_samples.shape[1]):
        Q = results.Q_samples[:,i].ravel()
        Q_xy = Q.reshape(Gx,Gy)
        mis[i] = mutual_information(Q_xy)

    # Compute mean and variance of the differential entropy
    mi_mean = np.mean(mis)
    mi_std= np.std(mis)
    results.mi_compute_time = time.clock() - entropy_start_time

# Estimate distribution of mutual information estimates
# Now do density estimation for the entropies
mi_G = 100
mi_box = [max(0., mi_mean - 5*mi_std), mi_mean + 5*mi_std]
mi_box[0] = min(mi_box[0],mi_true)
mi_box[1] = max(mi_box[1],mi_true)
mi_results = deft_1d.run(mis, G=mi_G, bbox=mi_box, alpha=3, periodic=False, num_samples=0)

#Q, R, Q_ub, Q_lb, Q_samples, map_curve
Q = results.Q_star
R = results.R
map_curve = results.map_curve

# Make Q 2d
Q_2d = Q.reshape(num_bins)
R_2d = R.reshape(num_bins)

# Calibrate both R and Q to the same scale
cl = [0.0, max(Q_true.ravel())]
cmap = plt.get_cmap('bone')

# Plot density
ax = plt.subplot(num_rows, num_cols, num_cols*row + 1)
plt.imshow(Q_true.T, interpolation='nearest', cmap=cmap)
plt.clim(cl)
plt.yticks([])
plt.xticks([])
plt.title("$Q_{true}$  (%s)"%data_type)

# Plot histogram
ax = plt.subplot(num_rows, num_cols, num_cols*row + 2)
plt.imshow(R_2d.T, interpolation='nearest', cmap=cmap)
plt.clim(cl)
plt.yticks([])
plt.xticks([])
plt.title("$R$ ($N = %d$)"%N)

# Plot density
ax = plt.subplot(num_rows, num_cols, num_cols*row + 3)
plt.imshow(Q_2d.T, interpolation='nearest', cmap=cmap)
plt.clim(cl)
plt.yticks([])
plt.xticks([])
plt.title("$Q_{star}$ ($\\alpha = %d$)"%alpha)

# Plot evidence ratio
log_Es = [p.log_E for p in map_curve.points]
ts = [p.t for p in map_curve.points]
t_star = ts[map_curve.i_star]
max_log_E = max(log_Es)
ax = plt.subplot(num_rows, num_cols, num_cols*row + 4)
plt.plot(ts[1:-1], log_Es[1:-1], '-o')
#plt.title("%d ts, t_star=%.1f, max(log_E)=%.1f"%(len(ts), t_star, max_log_E))

# Plot mutual information
ax = plt.subplot(num_rows, num_cols, num_cols*row + 5)
mi_h, mi_centers, mi_edges = utils.grid_info_from_bbox_and_G(mi_box, mi_G)
mi_R, _ = utils.histogram_counts_1d(mis, G=mi_G, bbox=mi_box, 
    normalized=True)
plt.bar(mi_edges[:-1], mi_R, width=mi_h, linewidth=0, color=gray, zorder=0)
plt.xlim(mi_box)

yl = [0, max(mi_results.Q_star)]
plt.plot([mi_true, mi_true], yl, '--k', linewidth=2)
plt.ylim(yl)

plt.plot(mi_results.bin_centers, mi_results.Q_star, 
    color=orange, linewidth=3)

z = (mi_true - mi_mean)/mi_std

#yl = [0, 1.2*max(eresults.Q_star)]
#plt.plot([true_e, true_e], yl, '--k', linewidth=2)
#plt.ylim(yl)
plt.yticks([])
plt.title("$%.3f \\pm %.3f$ bits ($z = %0.1f$)"%(mi_mean, mi_std, z))

if True:

    #
    # Open second window and plot samples
    #
    plt.figure(figsize=[15,7])
    results.num_samples
    num_rows = 5
    num_cols = 10
    K = num_rows*num_cols
    indices = np.random.choice(results.num_samples, size=K)
    for k,i in enumerate(indices):
        Q = results.Q_samples[:,i].ravel()
        Q_xy = Q.reshape(Gx,Gy)
        mi = mutual_information(Q_xy)
        plt.subplot(num_rows, num_cols, k+1)
        plt.imshow(Q_xy.T, interpolation='nearest', cmap=cmap)
        plt.clim(cl)
        plt.yticks([])
        plt.xticks([])
        #plt.title("I = %0.2f bits"%mi)

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.show()
#plt.tight_layout() # Needed so plot is drawn tollerably

# Save plot
#plt.savefig('report.test_maxent_2d.png')