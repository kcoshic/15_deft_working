import scipy as np
import numpy as np
import matplotlib.pyplot as plt

import deft_1d

plt.close('all')

# Colors to use
blue = [0.,0.,1.]
lightblue = [0.0, 0.5, 1.0]
orange = [1.,0.5,0.]
gray = [0.6, 0.6, 0.6]

bbox = [-5.,5.]
G = 300
alpha = 3
periodic=True
ell = 1.0

N = 10000
template_data = 2.*np.random.randn(N)

xgrid, Q, R = deft_1d.sample_from_deft_1d_prior(template_data, ell, G=G, alpha=alpha, bbox=bbox, periodic=periodic)

plt.figure()

h = xgrid[1]-xgrid[0]
left_bin_edges = xgrid-h/2.

plt.bar(left_bin_edges, R, width=h, linewidth=0, color=gray)

plt.plot(xgrid, Q)
plt.xlim(bbox)
plt.ylim([0, 1.5*max(Q)])

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
#plt.show()
#plt.tight_layout() # Needed so plot is drawn tollerably