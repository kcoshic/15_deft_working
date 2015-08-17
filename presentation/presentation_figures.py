import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# Add parent directory to path
import sys
sys.path.append('../code')
sys.path.append('../sim')

# Import deft modules
import deft_1d
import simulate_data_1d
import kde_1d

plt.ion()
plt.close('all')

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]


# Generate data
N = 100
G = 100
alpha = 3
data_type = 'wide'


# Create new data or load old data
new_data = True
if new_data:
    # Simulate data and get default deft settings
    data, defaults = simulate_data_1d.run(data_type,N)

    pickle.dump( (data,defaults), open( "data.p", "wb" ) )

else:
    data, defaults = pickle.load(open("data.p","rb"))

# Set bounding box
bbox = [defaults['box_min'], defaults['box_max']]
L = bbox[1]-bbox[0]
h = 1.*L/G

# Create a variety of histograms
num_bins = [5,20,100]

plt.figure()
for i, bins in enumerate(num_bins):
	ax = plt.subplot(len(num_bins),1,i+1)
	histogram, bin_edges = np.histogram(data,bins=bins,range=bbox,normed=True)
	h = bin_edges[1]-bin_edges[0]
	bin_centers = bin_edges[:-1]
	ax.bar(bin_centers, histogram, width=h, linewidth=0, color=gray)
	plt.yticks([])
	plt.ylim([0,max(histogram)*1.3])
plt.show()


assert False

# Plot kernel at each data point
def gaussian(x,mu,sigma):
	return sp.exp(-((x-mu)**2)/(2.*sigma**2))/sp.sqrt(2.*sp.pi*sigma**2)

# Do KDE
pdf, xs, kde_best = kde_1d.kde(data,G,bbox,details=True)

sigma = kde_best.bandwidth
plt.figure()
xs_fine = sp.linspace(bbox[0],bbox[1],1E4)
kernels = sp.zeros([len(xs_fine),num_samples])
for i, datum in enumerate(data):
	kernels[:,i] = gaussian(x=xs_fine,mu=datum,sigma=sigma)
plt.plot(xs_fine,kernels,'-b',alpha=0.2)
plt.plot(xs,pdf,'-b',linewidth=2)
plt.show()



# Do density estimation
results = deft_1d.run(data, G=G, alpha=alpha, bbox=bbox)
xs_hist = results.bin_edges[:-1]
xs = results.bin_centers
bin_centers = results.bin_centers
R = results.R
Q_star = results.Q_star
h = results.h
i_star = results.map_curve.i_star

# Plot histogram with density estimate on top
plt.figure(figsize=[3, 8])

# Set Qs along MAP curve to plot
Qs = [[], 
	  results.map_curve.points[0].Q,
	  results.map_curve.points[i_star - 20].Q,
	  results.map_curve.points[i_star].Q,
	  results.map_curve.points[i_star + 20].Q,
	  results.map_curve.points[-1].Q]

# Plot each Q
num_panels = len(Qs)
for i, Q in enumerate(Qs):
	ax = plt.subplot(num_panels,1,i+1)
	ax.bar(xs_hist, R, width=h, linewidth=0, color=gray, zorder=0)
	if len(Q)>0:
		ax.plot(xs, Q, color=blue, linewidth=2, zorder=1)
	plt.yticks([])
	plt.xticks([])


# # Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.show()
#plt.tight_layout() # Needed so plot is drawn tollerably

# # Save plot
plt.savefig('map_curve.png')


pdf = lambda x:  0.5*np.exp(-np.power(x-2.5,2)/2)/np.sqrt(2*np.pi) +               0.5*np.exp(-np.power(x+2.5,2)/2)/np.sqrt(2*np.pi)
yl = [0, 1.5*max(pdf(xs))]

plt.figure(figsize=[6, 6])
ax = plt.subplot(3,1,1)
ax.plot(xs, pdf(xs), color='k', linewidth=2, zorder=0)
plt.yticks([])
plt.ylim(yl)
#plt.xticks([])

ax = plt.subplot(3,1,2)
ax.bar(xs_hist, R, width=h, linewidth=0, color=gray, zorder=0)
plt.yticks([])
#plt.xticks([])

ax = plt.subplot(3,1,3)
ax.plot(xs, pdf(xs), color='k', linewidth=2, zorder=0)
ax.plot(xs, Q_star, color=blue, linewidth=2, zorder=1)
plt.yticks([])
plt.ylim(yl)
#plt.xticks([])

plt.savefig('the_problem.png')

# Plot histogram with density estimate on top
plt.figure(figsize=[5, 6])
# Plot each Q
num_panels = 2
for i, Q in enumerate(Qs[:2]):
	ax = plt.subplot(num_panels,1,i+1)
	ax.bar(xs_hist, R, width=h, linewidth=0, color=gray, zorder=0)
	if len(Q)>0:
		ax.plot(xs, Q, color=blue, linewidth=2, zorder=1)
	plt.yticks([])
	plt.xticks([])

plt.savefig('maxent.png')


# Plot log E vs ell
plt.figure(figsize=[5,3])
log_Es = [p.log_E for p in results.map_curve.points[1:-1]][::-1]
ells = [p.ell for p in results.map_curve.points[1:-1]][::-1]
plt.semilogx(ells,log_Es)
plt.xlim([min(ells), max(ells)])
plt.ylim([-1.2*max(log_Es), 1.2*max(log_Es)])
plt.show()
plt.savefig('log_E_vs_ell.png')
