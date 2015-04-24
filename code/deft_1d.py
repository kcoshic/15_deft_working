#!/usr/local/bin/python -W ignore
import os.path
import re
import argparse
import scipy as sp
import numpy as np
import sys
import time
#import gviz_api
import json

# Import deft-related code
import deft_core
import utils
import laplacian
import maxent

# Imports for KDE
import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV


class Results(): pass;


# Specify argument parser for DEFT and return arguments
def get_commandline_arguments():
    # Set up parser for commandline arguments
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument('-i', '--input', dest='data_file', default='stdin', type=str, help='File containing input data. Default: standard input')

    # Optional switch arguments
    parser.add_argument('-R', '--histogram', action='store_true', help='Include histogram in output file.')
    parser.add_argument('-M', '--max_ent', action='store_true', help='Include maximum entropy estimate in output file.')
    parser.add_argument('-p', '--periodic', action='store_true', help='Enforce periodic boundary conditions.')
    parser.add_argument('--json', action='store_true', help='Output data as a JSon string.')


    # Optional input arguments
    parser.add_argument('-a', '--alpha', type=int, default=3, choices=[1,2,3], help='Derivative to constrain.')
    parser.add_argument('-G', '--num_gridpoints', type=int, default=100, help='Number of gridpoints to use.')
    parser.add_argument('-e', '--epsilon', default=0.01, type=float, help='Geodesic distance spacing of computed densities along MAP curve.')
    parser.add_argument('-t', '--tollerance', default=0.000001, type=float, help='Tollerance for termination of corrector steps.')
    parser.add_argument('-N', '--num_samples', default=20, type=int, help='Number of samples to draw from the posterior density')
    parser.add_argument('--box_min', default=-sp.Inf, type=float)
    parser.add_argument('--box_max', default=sp.Inf, type=float)
    parser.add_argument('-c', '--map_curve_file', default='none', help='Name of MAP curve output file. Default: none.')
    parser.add_argument('-o', '--output_file', help='specify name for output file. Otherwise output is written to stdout')
    parser.add_argument('-l', '--length_scale', default='auto', help='Specify smoothness length scale: "auto" (default), "infinity", or a float.')
    parser.add_argument('--no_output', action='store_true', help="No not give any output. Useful for debugging only.")

    # Parse arguments
    args = parser.parse_args()

    # Return arguments
    return args

# Draws a probability distribution from the bilateral Laplacian prior
# You have to manually specify coefficients of nonconstant kernel components
# if alpha > 1
def sample_from_deft_1d_prior(template_data, ell, G=100, alpha=3, 
    bbox=[-np.Inf, np.Inf], periodic=False):

    # Create Laplacian
    if periodic:
        Delta = laplacian.Laplacian('1d_periodic', alpha, G, 1.0)
    else:
        Delta = laplacian.Laplacian('1d_bilateral', alpha, G, 1.0)

    # Get histogram counts and grid centers
    counts, bin_centers = utils.histogram_counts_1d(template_data, G, 
        bbox=bbox)
    R = 1.*counts/np.sum(counts)

    # Get other information agout grid
    bbox, h, bin_edges = utils.grid_info_from_bin_centers_1d(bin_centers)

    # Draw coefficients for other components of phi
    kernel_dim = Delta._kernel_dim
    kernel_basis = Delta._eigenbasis[:,:kernel_dim]
    rowspace_basis = Delta._eigenbasis[:,kernel_dim:]
    rowspace_eigenvalues = ell**(2*alpha) * h**(-2*alpha) * \
        np.array(Delta._eigenvalues[kernel_dim:]) 

    # Keep drawing coefficients until phi_rowspace is not minimized
    # at either extreme
    while True:

        # Draw coefficients for rowspace coefficients
        while True:
            rowspace_coeffs = \
                np.random.randn(G-kernel_dim)/np.sqrt(2.*rowspace_eigenvalues)

            # Construct rowspace phi
            rowspace_coeffs_col = np.mat(rowspace_coeffs).T
            rowspace_basis_mat = np.mat(rowspace_basis)
            phi_rowspace = rowspace_basis_mat*rowspace_coeffs_col

            #if not min(phi_rowspace) in phi_rowspace[[0,-1]]:
            break

        if kernel_dim == 1:
            phi_kernel = sp.zeros(phi_rowspace.shape)
            break

        # Construct full phi so that distribution mateches moments of R
        phi_kernel, success = maxent.compute_maxent_field(R, kernel_basis, 
            phi0=phi_rowspace, geo_dist_tollerance=1.0E-10)

        if success:
            break
        else:
            print 'Maxent failure! Trying to sample again.'

    phi_rowspace = np.array(phi_rowspace).ravel()
    phi_kernel = np.array(phi_kernel).ravel()
    phi = phi_kernel + phi_rowspace

    # Return Q
    Q = utils.field_to_prob(phi)/h
    R = R/h
    return bin_centers, Q, R

# Gets a handle to the data file, be it a system file or stdin
def get_data_file_handle(data_file_name):
    # Get data input file
    if data_file_name == 'stdin':
        data_file_handle = sys.stdin
        assert data_file_handle, 'Failed to open standard input.'

    else:
        # Verify that input file exists and that we have permission to read it
        assert os.path.isfile(data_file_name)

        # Open data file
        data_file_handle = open(data_file_name,'r')
        assert data_file_handle, 'Failed to open "%s"'%data_file_name

    # Return a handle to the data file
    return data_file_handle


# Loads data. Does requisite error checking
#### WARNING! Still need to modify to prevent reading too much information
def load_data(data_file_handle, MAX_NUM_DATA=1E6):

    # Read in data file. Discard all commented lines commented with '#' or '/'
    data_lines = [l.strip() for l in data_file_handle.readlines() if len(l) > 0 and not l[0] in ['#','/']]

    # Remove all non-numeric entities from lines, and form into one long string
    pattern = '[^0-9\-\+\.]|\-[^0-9\.]|\+[^0-9\.]|[^0-9]\.[^0-9]' # Removes all non-valid numbers
    data_string = ' '.join([re.sub(pattern, ' ', l) for l in data_lines])

    # Parse numbers from data_string
    data = sp.array([float(d) for d in data_string.split()])
    
    # Return data to user. 
    return data

#
# The main DEFT algorithm in 1D. 
#

def run(data, G=100, alpha=3, bbox=[-np.Inf, np.Inf], periodic=False, \
        resolution=3.14E-2, tollerance=1E-3, num_samples=10):
    
    # Start clock
    start_time = time.clock()

    # Create Laplacian
    if periodic:
        Delta = laplacian.Laplacian('1d_periodic', alpha, G, 1.0)
    else:
        Delta = laplacian.Laplacian('1d_bilateral', alpha, G, 1.0)

    # Get histogram counts and grid centers
    counts, bin_centers = utils.histogram_counts_1d(data, G, bbox=bbox)

    # Get other information agout grid
    bbox, h, bin_edges = utils.grid_info_from_bin_centers_1d(bin_centers)

    # Do DEFT density estimation
    Q_star, R, Q_ub, Q_lb, Q_samples, map_curve = deft_core.run(counts, Delta, resolution=resolution, tollerance=tollerance, details=True, errorbars=True, num_samples=num_samples)

    # Stop clock
    end_time = time.clock()

    # Build up list of results. 
    # Remember to rescale R and Q_star by 1/h so tha they are normalized
    results = Results()
    results.R = R/h
    results.Q_star = Q_star/h
    results.Q_lb = Q_lb/h
    results.Q_ub = Q_ub/h
    results.counts = counts
    results.N = sum(counts)
    results.bbox = bbox
    results.h = h
    results.G = G
    results.L = G*h
    results.bin_centers = bin_centers
    results.bin_edges = bin_edges
    results.resolution = resolution
    results.tollerance = tollerance
    results.periodic = periodic
    results.alpha = alpha
    results.execution_time = end_time-start_time
    results.map_curve = map_curve
    results.Q_samples = Q_samples/h

    return results
#
# Main program
#
if __name__ == '__main__':

    # Get commandline arguments using argparse
    args = get_commandline_arguments()

    # Get data file handle
    data_file_handle = get_data_file_handle(args.data_file)

    # Retrieve data from file
    data = load_data(data_file_handle)

    # Set alpha
    alpha = args.alpha

    # Set the number of gridpoints
    G = args.num_gridpoints

    # Make sure data spans a finite interval
    bbox = [args.box_min,args.box_max]

    # Get periodic switch
    periodic = args.periodic

    # Compute DEFT density estimate
    results = run(data, G=G, alpha=alpha, bbox=bbox, periodic=periodic, num_samples=args.num_samples)

    # Do KDE estimate as well
    #if args.kde:
    #    Q_kde = kde(data, G, bbox)
    
    # Output results
    Q_star = results.Q_star
    Q_lb = results.Q_lb
    Q_ub = results.Q_ub
    x_grid = results.bin_centers
    R = results.R
    Q_samples = [[q for q in Q] for Q in results.Q_samples.T]

    # Verify output
    assert len(Q_star)==G
    assert len(Q_ub)==G
    assert len(Q_lb)==G
    assert len(x_grid)==G
    assert len(R)==G
    #assert len(Q_kde) == G

    # Create dictionary to return as JSON object
    results_dict = {
        'Q_samples':Q_samples,
        'Q_star':list(Q_star),
        'Q_lb':list(Q_lb),
        'Q_ub':list(Q_ub),
        'x_grid':list(x_grid),
        'R':list(R),
        #'Q_kde':list(Q_kde),
        'h':results.h,
        'G':results.G,
        'L':results.L,
        'alpha':alpha,
        'box_min':bbox[0],
        'box_max':bbox[1],
        'periodic':periodic
    }

    # Output in JSon format
    if args.json:
        output_string = json.dumps(results_dict)

    # Tab-delimited text. 
    else:
        output_string = '\n'.join(['%f\t%f\t%f'%(x,r,q) for x,r,q\
                in zip(x_grid,R,Q_star)])

    # Write output
    if not args.no_output:
        sys.stdout.write(output_string+'\n')

    