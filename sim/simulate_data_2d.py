#!/usr/bin/python
import scipy as sp
import numpy as np
import argparse
import os.path
import scipy.stats as stats
import sys

import sys
sys.path.append('../code/')
sys.path.append('../sim/')
sys.path.append('../data/')
import digits

MAX_NUM_SAMPLES = 1E6

valid_distributions = '''
digit
digit_negative
uncorrelated_gaussian
correlated_gaussian
two_gaussians
uniform
three_corners
three_sharp_corners
four_corners
'''.split()

def digit(N, digit='random',background=0.0, negative=False):
    # Get image
    image = digits.get_digit_images(num=1, digit=digit)[0]
    image = image.T

    # Invert image if requested
    if negative:
        image_max = max(image.ravel())
        image = image_max - image

    # Convert image to list of lists
    Gx, Gy = image.shape
    image_list = [[image[i,j] for j in range(Gy)] for i in range(Gx)]

    # Have to add background, it seems, because of the numerics. Can I fix this?
    assert background >= 0.0
    bg = image.ravel().sum()*background/(1.*Gx*Gy)
    pdf_py = str(image_list) + '[int(x)][int(y)] + %f'%bg

    pdf_js = pdf_py

    box = ([-0.5, Gx-0.5], [-0.5, Gy-0.5])
    xs = range(Gx)
    ys = range(Gy)

    # Evaluate pdf at gridpoints
    ps = np.zeros(Gx*Gy)
    gridpoints = [(0,0)]*(Gx*Gy)
    for i in range(Gx):
        for j in range(Gy):
            x = xs[i] 
            y = ys[j]
            p = eval(pdf_py)
            if p < 0:
                print 'x = %f'%x
                print 'y = %f'%y
                print 'pdf_py: %s'%pdf_py
                print 'p == %f'%y
                assert False

            ps[i+Gx*j] = p
            gridpoints[i+Gx*j] = (x,y)
    ps /= sum(ps)

    # Sample datapoints
    indices = np.random.choice(Gx*Gy, size=N, replace=True, p=ps)

    # Reformat data: [(x,y)s] -> (xs, ys)
    data = ([gridpoints[i][0] for i in indices], 
        [gridpoints[i][1] for i in indices])

    return data, pdf_py, pdf_js, box


def gaussian_mixture(N,weights,mus,sigmas,rhos,box):
    assert len(weights)==len(mus)==len(sigmas)

    # Check input
    mus_x = np.array([m[0] for m in mus])
    mus_y = np.array([m[1] for m in mus])
    sigmas_x = np.array([s[0] for s in sigmas])
    sigmas_y = np.array([s[1] for s in sigmas])
    weights = np.array(weights)
    rhos = np.array(rhos)

    assert all(sigmas_x > 0)
    assert all(sigmas_y > 0)
    assert all(weights >= 0)

    # Get xs to sample
    Gx = 100
    Gy = 100
    xbox = box[0]
    ybox = box[1]
    assert xbox[0] < xbox[1]
    assert ybox[0] < ybox[1]
    xs = np.linspace(xbox[0],xbox[1],Gx)
    ys = np.linspace(ybox[0],ybox[1],Gy)

    # Build pdf strings
    pdf_py = '0'
    pdf_js = '0'
    for mx, my, sx, sy, r, w in zip(mus_x,mus_y,
                                    sigmas_x,sigmas_y,rhos,weights):
        pdf_py +='+(%f/(%f*%f))*np.exp('%(w,sx,sy)
        pdf_py += '-0.5*np.power((x-(%f))/%f,2)'%(mx,sx)
        pdf_py += '-0.5*np.power((y-(%f))/%f,2)'%(my,sy)
        pdf_py += '-%f*(x-(%f))*(y-(%f))/(%f*%f)'%(r,mx,my,sx,sy)
        pdf_py += ') '

        pdf_js +='+(%f/(%f*%f)*Math.exp('%(w,sx,sy)
        pdf_js += '-0.5*Math.pow((x-(%f))/%f,2)'%(mx,sx)
        pdf_js += '-0.5*Math.pow((y-(%f))/%f,2)'%(my,sy)
        pdf_js += '-%f*(x-(%f))*(y-(%f))/(%f*%f)'%(r,mx,my,sx,sy)
        pdf_js += ')'

    # Evaluate pdf at gridpoints
    ps = np.zeros(Gx*Gy)
    gridpoints = [(0,0)]*(Gx*Gy)
    for i in range(Gx):
        for j in range(Gy):
            x = xs[i] 
            y = ys[j]
            p = eval(pdf_py)
            if p < 0:
                print 'x = %f'%x
                print 'y = %f'%y
                print 'pdf_py: %s'%pdf_py
                print 'p == %f'%y
                assert False

            ps[i+Gx*j] = p
            gridpoints[i+Gx*j] = (x,y)
    ps /= sum(ps)

    # Sample datapoints
    indices = np.random.choice(Gx*Gy, size=N, replace=True, p=ps)

    # Reformat data: [(x,y)s] -> (xs, ys)
    data = ([gridpoints[i][0] for i in indices], 
        [gridpoints[i][1] for i in indices])

    # Return valuables. Note: data = (data_x_array, data_y_array)
    return data, pdf_py, pdf_js

# Specify argument parser for DEFT and return arguments
def get_commandline_arguments():
    # Set up parser for commandline arguments
    parser = argparse.ArgumentParser()

    # Group of functional forms to choose from
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--distribution', dest='distribution', \
        default='gaussian', choices=valid_distributions)

    # Number of data points to simulate
    parser.add_argument('-N', '--num_samples', dest='num_samples', type=int, \
        default=100, help='Number of data points to simulate.')

    # Output file, if any
    parser.add_argument('-o', '--output_file', default='stdout', \
        help='Specify where to write data to. Default: stdout')

    # Parse arguments
    args = parser.parse_args()

    # Add in defaults if necessary
    if args.distribution==None:
        args.distribution='gaussian'

    # Return fixed-up argument to user
    return args

def run(distribution_type='correlated_gaussian', N=100):
    """
    Performs the primary task of this module: simulating 2D data

    Args:
        - distribution_type (str): The distribution from which to draw data.
            must be one of the options listed in VALID_DISTRIBUTIONS.
        - N (int): The number of data points to simulate. Must be less than
            MAX_NUM_SAMPLES.

    Returns:
        - data (numpy.array): A length 2 tuple (x-data, y-data)
        - settings (dict): A dict object containing the settings
    """

    periodic = False
    pdf = ''
    alpha = 3

    if distribution_type == 'digit':
        description = 'Handwritten digit'
        data, pdf_py, pdf_js, box =  digit(N, digit='random')

    elif distribution_type == 'digit_negative':
        description = 'Handwritten digit, negative'
        data, pdf_py, pdf_js, box =  digit(N, digit='random', negative=True)

    elif distribution_type == 'uncorrelated_gaussian':
        description = 'Uncorrelated Gaussian'
        mus = [(0.,0.)]
        sigmas = [(1.,1.)]
        rhos = [0.]
        weights = [1.]
        box = ([-3,3],[-3,3])
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,
            sigmas,rhos,box)

    elif distribution_type == 'correlated_gaussian':
        description = 'Correlated Gaussian'
        mus = [(0.,0.)]
        sigmas = [(1.,1.)]
        rhos = [0.5]
        weights = [1.]
        box = ([-3,3],[-3,3])
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,
            sigmas,rhos,box)

    elif distribution_type == 'two_gaussians':
        description = 'Correlated Gaussian'
        mus = [(-2.5,0.),(2.5,2.)]
        sigmas = [(1.,1.),(1.,2.)]
        rhos = [0.,0.]
        weights = [1.,2.]
        box = ([-5,5],[-2.5,7.5])
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,
            sigmas,rhos,box)

    elif distribution_type == 'three_corners':
        description = 'Three corners'
        mus = [(0,0),(2.5,5),(5,0)]
        sigmas = [(1,1)]*3
        rhos = [0]*3
        weights = [1]*3
        box = ([-3,8],[-3,8])
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,
            sigmas,rhos,box)

    elif distribution_type == 'face':
        description = 'Face'
        mus = [(0,0), (2.5,6), (5,0)]
        sigmas = [(1,1), (2,1), (1,1)]
        rhos = [0, -0.6, 0]
        weights = [1, 2, 1]
        box = ([-3,9],[-3,9])
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,
            sigmas,rhos,box)

    elif distribution_type == 'three_sharp_corners':
        description = 'Three sharp corners'
        mus = [(0,0),(2.5,5),(5,0)]
        sigmas = [(0.5,0.5)]*3
        rhos = [0]*3
        weights = [1]*3
        box = ([-3,8],[-3,8])
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,
            sigmas,rhos,box)

    elif distribution_type == 'four_corners':
        description = 'Four corners'
        mus = [(0,0),(0,5),(5,5),(5,0)]
        sigmas = [(1,1)]*4
        rhos = [0]*4
        weights = [1]*4
        box = ([-3,8],[-3,8])
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,
            sigmas,rhos,box)
 
    elif distribution_type == 'uniform':
        description = 'Uniform'
        box = ([0,1],[0,1])
        alpha = 1
        a = stats.uniform.rvs(size=N)
        b = stats.uniform.rvs(size=N)
        data = (a,b)
        pdf_py = '1.0'
        pdf_js = '1.0'

    else:
        print 'Distribution type "%s" not recognized.'%distribution_type
        raise

    settings = {
        'box_xmin':box[0][0],
        'box_xmax':box[0][1],
        'box_ymin':box[1][0],
        'box_ymax':box[1][1],
        'alpha':alpha, 
        'periodic':periodic, 
        'N':N,
        'description':description,
        'pdf_py':pdf_py,
        'pdf_js':pdf_js
    }

    # Data returned is a length 2 tuple (x-data, y-data)
    return data, settings

#
# Main program
#

if __name__ == '__main__':

    # Get commandline arguments
    args = get_commandline_arguments()

    # Make sure number of data points is reasonable
    N = args.num_samples
    assert(N == int(N))
    assert(N > 0)
    assert(N <= MAX_NUM_SAMPLES)

    # Generate data
    data = generate_data(args.distribution, N)

    # Format data as string
    data_string = '\n'.join(['%f,\t%f,'%d for d in data])+'\n'

    # Set output stream
    if args.output_file=='stdout':
        out_stream = sys.stdout
    else:
        out_stream = open(args.output_file,'w')
        assert out_stream, 'Failed to open file "%s" for writing.'%args.output_file

    # Write data to stdout
    out_stream.write(data_string)

    # Close output stream
    out_stream.close()


