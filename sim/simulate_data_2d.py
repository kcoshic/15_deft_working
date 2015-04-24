#!/usr/bin/python
import scipy as sp
import argparse
import os.path
import scipy.stats as stats
import sys

MAX_NUM_SAMPLES = 1E6

valid_distributions = '''
gaussian
two_gaussians
uniform
beta_convex
beta_concave
beta_saddle
exponential
exponential_uniform
gamma
pyramid
triangular
laplace
vonmises
'''.split()

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

def generate_data(distribution_type='gaussian', N=100):

	# If gaussian distribution
	if distribution_type == 'gaussian':
		data_x = stats.norm.rvs(size=N)
		data_y = stats.norm.rvs(size=N)

	# If gaussian distribution
	elif distribution_type == 'two_gaussians':
		x1 = stats.norm.rvs(size=N)
		y1 = stats.norm.rvs(size=N)
		x2 = 5.0 + stats.norm.rvs(size=N)
		y2 = 2.0 + 2.0*stats.norm.rvs(size=N)

		r = stats.uniform.rvs(size=N)
		indices = r < .66666

		data_x = x1
		data_x[indices] = x2[indices] 
		data_y = y1
		data_y[indices] = y2[indices] 

	# If uniform distribution 
	elif distribution_type == 'uniform':
		data_x = stats.uniform.rvs(size=N)
		data_y = stats.uniform.rvs(size=N)

	# Convex beta distribution 
	elif distribution_type == 'beta_convex':
		data_x = stats.beta.rvs(a=0.5, b=0.5, size=N)
		data_y = stats.beta.rvs(a=0.5, b=0.5, size=N)

	# Concave beta distribution 
	elif distribution_type == 'beta_concave':
		data_x = stats.beta.rvs(a=2, b=2, size=N)
		data_y = stats.beta.rvs(a=2, b=2, size=N)

	# Concave beta distribution 
	elif distribution_type == 'beta_saddle':
		data_x = stats.beta.rvs(a=0.5, b=0.5, size=N)
		data_y = stats.beta.rvs(a=2, b=2, size=N)

	# Exponential distribution
	elif distribution_type == 'exponential':
		data_x = stats.expon.rvs(size=N)
		data_y = stats.expon.rvs(size=N)

	# Exponential distribution
	elif distribution_type == 'exponential_uniform':
		data_x = stats.expon.rvs(size=N)
		data_y = stats.uniform.rvs(size=N)

	# Gamma distribution
	elif distribution_type == 'gamma':
		data_x = stats.gamma.rvs(a=3, size=N)
		data_y = stats.gamma.rvs(a=1, size=N)

	# Pyramid distribution
	elif distribution_type == 'pyramid':
		data_x = stats.triang.rvs(c=0.5, size=N)
		data_y = stats.triang.rvs(c=0.5, size=N)

	# Pyramid distribution
	elif distribution_type == 'triangular':
		data_x = stats.triang.rvs(c=0.5, size=N)
		data_y = stats.uniform.rvs(size=N)

	# Laplace distribution
	elif distribution_type == 'laplace':
		data_x = stats.laplace.rvs(size=N)
		data_y = stats.laplace.rvs(size=N)

	# von Misses distribution
	elif distribution_type == 'vonmises':
		data_x = stats.vonmises.rvs(kappa=1)
		data_y = stats.vonmises.rvs(kappa=1)

	else:
		print 'Distribution type "%s" not recognized.'%distribution_type
		raise

	return zip(data_x, data_y)

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


