#!/usr/local/bin/python
import os.path
import re
import argparse
import scipy as sp
import numpy as np
import sys
import time
import json

# Imports for KDE
import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

# Import DEFT utils. Just used for histogramming
import utils

class Results(): pass;

# Specify argument parser for DEFT and return arguments
def get_commandline_arguments():

    # Set up parser for commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='data_file', default='stdin', type=str, help='File containing input data. Default: standard input')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--json', action='store_true', help='Output data as a JSon string.')
    parser.add_argument('-G', '--num_gridpoints', type=int, default=100, help='Number of gridpoints to use.')
    parser.add_argument('--box_min', default=-sp.Inf, type=float)
    parser.add_argument('--box_max', default=sp.Inf, type=float)
    parser.add_argument('-o', '--output_file', help='specify name for output file. Otherwise output is written to stdout')
    parser.add_argument('--no_output', action='store_true', help="No not give any output. Useful for debugging only.")
    parser.add_argument('-p', '--periodic', action='store_true', help='Enforce periodic boundary conditions.')

    # Parse arguments
    args = parser.parse_args()

    # Return arguments
    return args


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
# The main KDE algorithm
#

# Kernel density esitmation function
def kde(data, G, bbox, cv=20, details=False):

    # Define grid
    h, bin_centers, bin_edges = utils.grid_info_from_bbox_and_G(bbox,G)
    L = bbox[1]-bbox[0]
    
    # Crop data
    indices = (data > bbox[0]) & (data < bbox[1])
    data = data[indices]
    N = len(data)

    # Use cv-fold cross-validation. Max 10-fold
    cv = min(N,cv)
    
    # Set bandwidths to check
    bandwidths = np.logspace(np.log10(h),np.log10(L),30)
    
    # Do KDE
    grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': bandwidths},
                    cv=cv) # cv-fold cross-validation
    grid.fit(data[:, None])
    
    # Retrieve best KDE estimate
    kde_best = grid.best_estimator_
    
    # Compute, normalize, and return pdf
    pdf = np.exp(kde_best.score_samples(bin_centers[:, None]))
    pdf /= h*np.sum(pdf)
    if not details:
        return pdf
    else:
        return pdf, bin_centers, kde_best

def run(data, G=100, bbox=[-np.Inf, np.Inf]):
    
    # Start clock
    start_time = time.clock()

    # Get histogram counts and grid centers
    counts, bin_centers = utils.histogram_counts_1d(data, G, bbox=bbox)
    R = 1.*counts/np.sum(counts)

    # Get other information agout grid
    bbox, h, bin_edges = utils.grid_info_from_bin_centers_1d(bin_centers)

    # Do density estimation
    Q_kde = kde(data, G, bbox)

    # Stop clock
    end_time = time.clock()

    # Build up list of results. 
    # Remember to rescale R and Q_star by 1/h so tha they are normalized
    results = Results()
    results.R = R/h
    results.Q_kde = Q_kde # ALREADY PROPERLY NORMALIZED!
    results.counts = counts
    results.N = np.sum(counts)
    results.bbox = bbox
    results.h = h
    results.G = G
    results.L = G*h
    results.bin_centers = bin_centers
    results.bin_edges = bin_edges

    return results
#
# Main program
#
if __name__ == '__main__':

    #
    # Grab arguments
    #

    # Get commandline arguments using argparse
    args = get_commandline_arguments()

    # Get data file handle
    data_file_handle = get_data_file_handle(args.data_file)

    # Retrieve data from file
    data = load_data(data_file_handle)

    # Set the number of gridpoints
    G = args.num_gridpoints

    # Make sure data spans a finite interval
    bbox = [args.box_min,args.box_max]

    #
    # Do density estimation
    #

    # Run deft
    results = run(data, G=G, bbox=bbox)

    #
    # Output results
    #

    # Create dictionary to return as JSON object
    results_dict = {
        'Q_kde':list(results.Q_kde),
        'x_grid':list(results.bin_centers),
        'R':list(results.R),
        'h':results.h,
        'G':results.G,
        'L':results.L,
        'box_min':results.bbox[0],
        'box_max':results.bbox[1]
    }

    # Output in JSon format
    if args.json:
        output_string = json.dumps(results_dict)

    # Tab-delimited text. 
    else:
        output_string = '\n'.join(['%f\t%f\t%f'%(x,r,q) for x,r,q\
                in zip(results.bin_centers,results.R,results.Q_kde)])

    # Write output
    if not args.no_output:
        sys.stdout.write(output_string+'\n')

    
