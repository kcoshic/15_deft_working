#!/usr/bin/python
import argparse
import json
import numpy as np
import math
import scipy as sp
import scipy.stats as stats
import sys


# List of supported distributions by name
VALID_DISTRIBUTIONS = '''
gaussian
narrow
wide
foothills
accordian
goalposts
towers
uniform
beta_convex
beta_concave
exponential
gamma
triangular
laplace
vonmises
'''.split()

# Maximum number of samples this algorithm will simulate
MAX_NUM_SAMPLES = 1E6

class Results(): pass;

def get_commandline_arguments():
    """ Specifies commandline arguments for simulate_data_1d.py """

    # Set up parser for commandline arguments
    parser = argparse.ArgumentParser()

    # Which distribution to choose from
    parser.add_argument('-d', '--distribution', dest='distribution', 
        default='gaussian', choices=VALID_DISTRIBUTIONS)

    # Number of data points to simulate
    parser.add_argument('-N', '--num_samples', dest='num_samples', type=int, 
        default=100, help='Number of data points to simulate.')

    # Output file, if any
    parser.add_argument('-o', '--output_file', default='stdout', 
        help='Specify where to write data to. Default: stdout')

    # Output data in JSON format if requested
    parser.add_argument('-j', '--json', action='store_true', 
        help='Output data as a JSON string.')

    # Parse arguments
    args = parser.parse_args()

    # Add in defaults if necessary
    if args.distribution==None:
        args.distribution='gaussian'

    # Return fixed-up argument to user
    return args

def gaussian_mixture(N,weights,mus,sigmas,bbox):
    assert bbox[1] > bbox[0]
    assert len(weights)==len(mus)==len(sigmas)

    # Get xs to sample
    xs = np.linspace(bbox[0],bbox[1],1E4)

    # Build pdf strings
    pdf_py = '0'
    pdf_js = '0'
    for m, s, w in zip(mus,sigmas,weights):
        pdf_py +='+(%f/%f)*np.exp(-0.5*np.power((x-(%f))/%f,2))'%(w,s,m,s)
        pdf_js +='+(%f/%f)*Math.exp(-0.5*Math.pow((x-(%f))/%f,2))'%(w,s,m,s)

    # Evaluate pdf at xs
    ps = np.zeros(len(xs))
    for i,x in enumerate(xs):
        ps[i] = eval(pdf_py)
    ps /= sum(ps)

    # Sample datapoints
    data = np.random.choice(xs, size=N, replace=True, p=ps)

    # Return valuables
    return data, pdf_py, pdf_js


def run(distribution_type='gaussian', N=100):
    """
    Performs the primary task of this module: simulating 1D data

    Args:
        - distribution_type (str): The distribution from which to draw data.
            must be one of the options listed in VALID_DISTRIBUTIONS.
        - N (int): The number of data points to simulate. Must be less than
            MAX_NUM_SAMPLES.

    Returns:
        - data (numpy.array): An array of N data points drawn from the 
            specified distribution
        - settings (dict): A dict object containing the settings
    """

    periodic = False
    alpha = 3

    # If gaussian distribution
    if distribution_type == 'gaussian':
        description = 'Gaussian distribution'
        mus = [0.]
        sigmas = [1.]
        weights = [1.]
        bbox = [-5,5]
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,sigmas,bbox)

    # If mixture of two gaussian distributions
    elif distribution_type == 'narrow':
        description = 'Gaussian mixture, narrow separation'
        mus = [-1.25, 1.25]
        sigmas = [1., 1.]
        weights = [1., 1.]
        bbox = [-6, 6]
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,sigmas,bbox)

    # If mixture of two gaussian distributions
    elif distribution_type == 'wide':
        description = 'Gaussian mixture, wide separation'
        mus = [-2.5, 2.5]
        sigmas = [1., 1.]
        weights = [1., 1.]
        bbox = [-6, 6]
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,sigmas,bbox)

    elif distribution_type == 'foothills':
        description = 'Foothills (Gaussian mixture)'
        mus = [0., 5., 8., 10, 11]
        sigmas = [2., 1., 0.5, 0.25, 0.125]
        weights = [1., 1., 1., 1., 1.]
        bbox = [-5,12]
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,sigmas,bbox)

    elif distribution_type == 'accordian':
        description = 'Accordian (Gaussian mixture)'
        mus = [0., 5., 8., 10, 11, 11.5]
        sigmas = [2., 1., 0.5, 0.25, 0.125, 0.0625]
        weights = [16., 8., 4., 2., 1., 0.5]
        bbox = [-5,13]
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,sigmas,bbox)

    elif distribution_type == 'goalposts':
        description = 'Goalposts (Gaussian mixture)'
        mus = [-20, 20]
        sigmas = [1., 1.]
        weights = [1., 1.]
        bbox = [-25,25]
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,sigmas,bbox)

    elif distribution_type == 'towers':
        description = 'Towers (Gaussian mixture)'
        mus = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        sigmas = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        weights = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
        bbox = [-25,25]
        data, pdf_py, pdf_js = gaussian_mixture(N,weights,mus,sigmas,bbox)

    # If uniform distribution   
    elif distribution_type == 'uniform':
        data = stats.uniform.rvs(size=N)
        bbox = [0,1]
        description = 'Uniform distribution'
        pdf_js = "1.0"
        pdf_py = "1.0"

    # Convex beta distribution
    elif distribution_type == 'beta_convex':
        data = stats.beta.rvs(a=0.5, b=0.5, size=N)
        bbox = [0,1]
        description = 'Convex beta distribtuion'
        pdf_js = "Math.pow(x,-0.5)*Math.pow(1-x,-0.5)*math.gamma(1)/(math.gamma(0.5)*math.gamma(0.5))"
        pdf_py = "np.power(x,-0.5)*np.power(1-x,-0.5)*math.gamma(1)/(math.gamma(0.5)*math.gamma(0.5))"

    # Concave beta distribution
    elif distribution_type == 'beta_concave':
        data = stats.beta.rvs(a=2, b=2, size=N)
        bbox = [0,1]
        description = 'Concave beta distribution'
        pdf_js = "Math.pow(x,1)*Math.pow(1-x,1)*math.gamma(4)/(math.gamma(2)*math.gamma(2))"
        pdf_py = "np.power(x,1)*np.power(1-x,1)*math.gamma(4)/(math.gamma(2)*math.gamma(2))"

    # Exponential distribution
    elif distribution_type == 'exponential':
        data = stats.expon.rvs(size=N)
        bbox = [0,5]
        description = 'Exponential distribution'
        pdf_js = "Math.exp(-x)"
        pdf_py = "np.exp(-x)"

    # Gamma distribution
    elif distribution_type == 'gamma':
        data = stats.gamma.rvs(a=3, size=N)
        bbox = [0,10]
        description = 'Gamma distribution'
        pdf_js = "Math.pow(x,2)*Math.exp(-x)/math.gamma(3)"
        pdf_py = "np.power(x,2)*np.exp(-x)/math.gamma(3)"

    # Triangular distribution
    elif distribution_type == 'triangular':
        data = stats.triang.rvs(c=0.5, size=N)
        bbox = [0,1]
        description = 'Triangular distribution'
        pdf_js = "2-4*Math.abs(x - 0.5)"
        pdf_py = "2-4*np.abs(x - 0.5)"

    # Laplace distribution
    elif distribution_type == 'laplace':
        data = stats.laplace.rvs(size=N)
        bbox = [-5,5]
        description = "Laplace distribution"
        pdf_js = "0.5*Math.exp(- Math.abs(x))"
        pdf_py = "0.5*np.exp(- np.abs(x))"

    # von Misses distribution
    elif distribution_type == 'vonmises':
        data = stats.vonmises.rvs(1, size=N)
        bbox = [-3.14159,3.14159]
        periodic = True
        description = 'von Mises distribution'
        pdf_js = "Math.exp(Math.cos(x))/7.95493"
        pdf_py = "np.exp(np.cos(x))/7.95493"

    else:
        assert False, 'Distribution type "%s" not recognized.'% \
            distribution_type

    settings = {
        'box_min':bbox[0],
        'box_max':bbox[1],
        'alpha':alpha, 
        'periodic':periodic, 
        'N':N,
        'description':description,
        'pdf_js':pdf_js,
        'pdf_py':pdf_py
    }

    return data, settings


def main():
    """ Commandline functionality of module. """

    # Get commandline arguments
    args = get_commandline_arguments()

    # Make sure number of data points is reasonable
    N = args.num_samples
    assert N == int(N)
    assert N > 0
    assert N <= MAX_NUM_SAMPLES

    # Generate data
    data, settings = run(args.distribution, N)

    # Set output stream
    if args.output_file=='stdout':
        out_stream = sys.stdout
    else:
        out_stream = open(args.output_file,'w')
        assert out_stream, \
            'Failed to open file "%s" for writing.'%args.output_file

    # If requested, output json format
    if args.json:

        # Create dictionary to hold all output information
        output_dict = settings
        output_dict['data'] = list(data)

        # Create JSON string and write to output
        json_string = json.dumps(output_dict)
        out_stream.write(json_string)

    else:
        # Format data as string
        data_string = '\n'.join(['%f'%d for d in data])+'\n'

        # Write bbox to stdout
        out_stream.write('# box_min: %f\n'%settings['box_min'])
        out_stream.write('# box_max: %f\n'%settings['box_max'])
        out_stream.write('# alpha: %d\n'%settings['alpha'])
        out_stream.write('# periodic: %s\n'%str(settings['periodic']))

        # Write data to stdout
        out_stream.write(data_string)

    # Close output stream
    out_stream.close()

# Executed if run at the commandline
if __name__ == '__main__':
    main()

