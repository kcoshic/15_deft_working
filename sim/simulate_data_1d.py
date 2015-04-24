#!/usr/bin/python
import argparse
import json
import numpy as np
import scipy as sp
import scipy.stats as stats
import sys


# List of supported distributions by name
VALID_DISTRIBUTIONS = '''
gaussian
narrow
wide
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


def run(distribution_type='gaussian', N=100):
    """
    Performs the primary task of this module: simulating data

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
    pdf = ''

    # If gaussian distribution
    if distribution_type == 'gaussian':
        #data = sp.randn(N)
        data = stats.norm.rvs(size=N)
        bbox = [-5,5]
        alpha = 3
        description = 'Gaussian distribution'
        pdf = "Math.exp(-Math.pow(x,2)/2)/Math.sqrt(2*Math.PI)"

    # If mixture of two gaussian distributions
    elif distribution_type == 'narrow':
        N1 = sp.floor(N/2);
        N2 = N - N1;
        separation = 2.5 
        data1 = stats.norm.rvs(size=N1) - separation/2.0
        data2 = stats.norm.rvs(size=N2) + separation/2.0
        data = sp.concatenate((data1,data2))
        bbox = [-6,6]
        alpha = 3
        description = 'Gaussian mixture, %s separation'%\
            distribution_type
        pdf = " 0.5*Math.exp(-Math.pow(x-1.25,2)/2)/Math.sqrt(2*Math.PI) + \
                0.5*Math.exp(-Math.pow(x+1.25,2)/2)/Math.sqrt(2*Math.PI) "

    # If mixture of two gaussian distributions
    elif distribution_type == 'wide':
        N1 = sp.floor(N/2);
        N2 = N - N1;
        separation = 5.0 
        data1 = stats.norm.rvs(size=N1) - separation/2.0
        data2 = stats.norm.rvs(size=N2) + separation/2.0
        data = sp.concatenate((data1,data2))
        bbox = [-6,6]
        alpha = 3
        description = 'Gaussian mixture, %s separation'%\
            distribution_type
        pdf = " 0.5*Math.exp(-Math.pow(x-2.5,2)/2)/Math.sqrt(2*Math.PI) + \
              0.5*Math.exp(-Math.pow(x+2.5,2)/2)/Math.sqrt(2*Math.PI) "

    # If uniform distribution   
    elif distribution_type == 'uniform':
        data = stats.uniform.rvs(size=N)
        bbox = [0,1]
        alpha = 1
        description = 'Uniform distribution'
        pdf = "1.0"

    # Convex beta distribution
    elif distribution_type == 'beta_convex':
        data = stats.beta.rvs(a=0.5, b=0.5, size=N)
        bbox = [0,1]
        alpha = 1
        description = 'Convex beta distribtuion'
        pdf = "Math.pow(x,-0.5)*Math.pow(1-x,-0.5)*math.gamma(1)/(math.gamma(0.5)*math.gamma(0.5))"

    # Concave beta distribution
    elif distribution_type == 'beta_concave':
        data = stats.beta.rvs(a=2, b=2, size=N)
        bbox = [0,1]
        alpha = 1
        description = 'Concave beta distribution'
        pdf = "Math.pow(x,1)*Math.pow(1-x,1)*math.gamma(4)/(math.gamma(2)*math.gamma(2))"

    # Exponential distribution
    elif distribution_type == 'exponential':
        data = stats.expon.rvs(size=N)
        bbox = [0,5]
        alpha = 2
        description = 'Exponential distribution'
        pdf = "Math.exp(-x)"

    # Gamma distribution
    elif distribution_type == 'gamma':
        data = stats.gamma.rvs(a=3, size=N)
        bbox = [0,10]
        alpha = 3
        description = 'Gamma distribution'
        pdf = "Math.pow(x,2)*Math.exp(-x)/math.gamma(3)"

    # Triangular distribution
    elif distribution_type == 'triangular':
        data = stats.triang.rvs(c=0.5, size=N)
        bbox = [0,1]
        alpha = 1
        description = 'Triangular distribution'
        pdf = "2-4*Math.abs(x - 0.5)"

    # Laplace distribution
    elif distribution_type == 'laplace':
        data = stats.laplace.rvs(size=N)
        bbox = [-5,5]
        alpha = 1
        description = "Laplace distribution"
        pdf = "0.5*Math.exp(- Math.abs(x))"

    # von Misses distribution
    elif distribution_type == 'vonmises':
        data = stats.vonmises.rvs(1, size=N)
        bbox = [-3.14159,3.14159]
        periodic = True
        alpha = 3
        description = 'von Mises distribution'
        pdf = "Math.exp(Math.cos(x))/7.95493"

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
        'pdf':pdf
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

