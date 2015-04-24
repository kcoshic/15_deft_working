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

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]

# Generate data
N = 100
G = 100
alpha = 3

data_type = 'gaussian'

# Compute data set
data, defaults = simulate_data_1d.run(data_type,N)

# Do density estimation
results = deft_1d.run(data, G=G, alpha=alpha, \
		bbox=defaults.bbox, \
		periodic=defaults.periodic)

# Plot the MAP curve
map_curve =results.map_curve
points = map_curve.get_points()

ts = sp.array([p.t for p in points])

num_points = len(points)
Qs = sp.zeros([G,num_points])
for n, p in enumerate(points):
	Qs[:,n] = p.Q

log_Es, ts = map_curve.get_log_evidence_ratios(finite=True)

plt.figure()
plt.ion()

plt.subplot(2,1,1)
plt.imshow(Qs.T, interpolation='nearest')

plt.subplot(2,1,2)
plt.plot(ts,log_Es)
y_max = max(log_Es)
y_min = min(log_Es)
plt.ylim([y_min, y_max])

plt.show()