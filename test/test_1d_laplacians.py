import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.append('../')

# Import deft modules
import laplacian
import utils

# Check orthonormality of eigenvectors

plt.figure(figsize=[ 10.6125, 9.8625])

file_names = [
    '1d_bilateral_alpha_1_G_300.pickle',
    '1d_bilateral_alpha_2_G_300.pickle',
    '1d_bilateral_alpha_3_G_300.pickle',
    '1d_periodic_alpha_3_G_300.pickle']
directory = 'laplacians/'

num_rows = len(file_names)
num_cols = 4
for a, file_name in enumerate(file_names):
	
	print 'working on row %d...'%(a+1)
	
	Delta = laplacian.load(directory+file_name)

	vs = Delta._eigenbasis
	alpha = Delta._order
	G = Delta._G
	op_type = Delta._type

	basis_dots = \
	   [[utils.dot(vs[:,i],vs[:,j]) for i in range(G)] for j in range(G)]
        basis_dots = sp.array(basis_dots)

	plt.subplot(num_rows, num_cols, num_cols*a+1)
	plt.imshow(basis_dots,interpolation='nearest')
	plt.clim([-.5,1.5])
	plt.title(r'%s $\alpha$=%d '%(op_type, alpha))
	
	plt.subplot(num_rows, num_cols, num_cols*a+2)
	diag_indices = (np.identity(G) == 1)
	off_diag_indices = (np.identity(G) == 0)
	diag_dots = basis_dots[diag_indices]
	off_diag_dots = basis_dots[off_diag_indices]
	ind = [0,1]
	ymean = [sp.mean(diag_dots), sp.std(off_diag_dots)]
	ystd = [sp.std(diag_dots), sp.std(off_diag_dots)]
	plt.bar(ind, ymean, 0.9, yerr=ystd)
	plt.title('on/off-diag dots')
	plt.ylim([-.2,1.2])
	plt.xlim([-.2,2.2])
	plt.xticks([.5, 1.5], ['on', 'off'])

	plt.subplot(num_rows, num_cols, num_cols*a+3)
	ys = Delta._eigenvalues**(1./(2.*alpha))
	plt.plot(ys)
	plt.title('eigenvalues')

	plt.subplot(num_rows, num_cols, num_cols*a+4)
	plt.plot(Delta._kernel_basis)
        plt.ylim([-3,3])
        plt.title('kernel basis')

# Plotting incantation
plt.ion() # So focus goes back to commandline
plt.draw() # Needed to avoid "CGContextRef is NULL" exception
plt.tight_layout() # Needed so plot is drawn tollerably

# Save plot
plt.savefig('report.test_1d_laplacians.png')

