import scipy as sp
import numpy as np

from numpy.polynomial.legendre import legval, legval2d

# A very small floating point number, used to prevent taking logs of 0
TINY_FLOAT64 = sp.finfo(sp.float64).tiny
TINY_FLOAT32 = sp.finfo(sp.float32).tiny
PHI_MIN = -50
PHI_MAX = 50

# This is useful for testing whether something is a number
NUMBER = (int, float, long)
ARRAY = (np.ndarray, list)

# Evaluate geodesic distance 
def geo_dist(P,Q):

    # Make sure these are valid distributions
    assert all(np.isreal(P))
    assert all(P >= 0)
    assert any(P > 0)
    assert all(np.isreal(Q))
    assert all(Q >= 0)
    assert any(Q > 0)

    # Enforce proper normalization
    P_prob = P/sp.sum(P) 
    Q_prob = Q/sp.sum(Q) 

    # Return geodistance. arccos can behave badly if argument
    # is too close to one, so prepare for this
    try:
        dist = 2*sp.arccos(sp.sum(sp.sqrt(P_prob*Q_prob)))
        assert np.isreal(dist)
    except:
        if sp.sum(sp.sqrt(P_prob*Q_prob)) > 1 - TINY_FLOAT32:
            dist = 0
        else:
            raise
    return dist

# Convert field to non-normalized probabiltiy distribution
def field_to_quasiprob(phi, regularize=False):

    if regularize:
        phi[phi < PHI_MIN] = PHI_MIN 
        phi[phi > PHI_MAX] = PHI_MAX 

    try:
        assert all(np.isreal(phi))
    except:
        print phi
        raise

    G = 1.*len(phi)

    quasiQ = sp.exp(-phi)/(1.*G)

    try:
        assert all(np.isfinite(quasiQ))
    except:
        print phi
        print quasiQ
        raise
    # Replace problematic values with zero
    #indices = ~np.isfinite(quasiQ)
    #quasiQ[indices] = 0.0

    # Check to see if Q has been fixed. If not, raise error
    assert all(np.isreal(quasiQ))
    if not all(quasiQ) >= 0:
        print 'Error: Q cannot be calculated from phi'
        raise

    # Return quasiprob
    return quasiQ

# Convert field to normalized probability distribution
def field_to_prob(phi, regularize=False):  
    quasiQ = field_to_quasiprob(phi, regularize=regularize)
    Q = quasiQ/sum(quasiQ)
    return Q

# Convert probability distribution to field
def prob_to_field(Q):
    G = len(Q)
    assert(all(np.isreal(Q)))
    assert(all(Q >= 0))
    phi = -sp.log(G*Q + TINY_FLOAT64)
    assert(all(np.isreal(phi)))
    return phi

def grid_info_from_bin_centers_1d(bin_centers):
    bin_centers = np.array(bin_centers)
    h = bin_centers[1]-bin_centers[0]
    bbox = [bin_centers[0]-h/2., bin_centers[-1]+h/2.]
    G = len(bin_centers)
    bin_edges = np.zeros(G+1)
    bin_edges[0] = bbox[0]
    bin_edges[-1] = bbox[1]
    bin_edges[1:-1] = bin_centers[:-1]+h/2.
    return bbox, h, bin_edges

def grid_info_from_bin_edges_1d(bin_edges):
    bin_edges = np.array(bin_edges)
    h = bin_edges[1]-bin_edges[1]
    bbox = [bin_edges[0], bin_edges[-1]]
    G = len(bin_centers)
    bin_centers = bin_edges[:-1]+h/2.
    return bbox, h, bin_centers

def grid_info_from_bbox_and_G(bbox, G):
    bin_edges = np.linspace(bbox[0], bbox[1], num=G+1, endpoint=True)
    h = bin_edges[1]-bin_edges[0]
    bin_centers = bin_edges[:-1]+h/2.
    return h, bin_centers, bin_edges


# Make a 1d histogram. Bounding box is optional
def histogram_counts_1d(data, G, bbox=[-np.Inf,np.Inf]):

    # Get interval spanned by data
    data_interval = max(data)-min(data)

    # Make sure that this interval is finite
    assert np.isfinite(data_interval)

    # Make sure the data actually have some spread
    assert data_interval > 0

    # Make sure that bbox is actually the right side
    assert len(bbox) == 2

    # Set lower bound automatically if called for
    if bbox[0] == -np.Inf:
        bbox[0] = min(data) - data_interval*0.2
    assert isinstance(bbox[0],NUMBER)

    # Set upper bound automatically if called for
    if bbox[1] == np.Inf:
        bbox[1] = max(data) + data_interval*0.2
    assert isinstance(bbox[1],NUMBER)

    # Make sure the lower obund is less than upper bound!
    assert bbox[0] < bbox[1]

    # Crop data to bounding box 
    indices = (data > bbox[0]) & (data < bbox[1])
    cropped_data = data[indices]

    # Get grid info from bbox and G
    h, bin_centers, bin_edges = grid_info_from_bbox_and_G(bbox, G)

    # Get counts in each bin
    counts, _ = np.histogram(data, bins=bin_edges, density=False)

    # Return the number of counts, the bin centers, and the bbox
    return counts, bin_centers

# Make a histogram
def histogram_2d(data, num_bins=[10,10], normalized=False):
    data_x = [d[0] for d in data]
    data_y = [d[1] for d in data]
    hist, xedges, yedges = np.histogram2d(data_x, data_y, bins=num_bins,\
        normed=normalized)
    hx = xedges[1]-xedges[0]
    hy = yedges[1]-yedges[1]

    xs =xedges[:-1]-hx/2.
    ys =yedges[:-1]-hx/2.
    return hist, xs, ys

# Returns the left edges of a binning given the centers
def left_edges_from_centers(centers):
    h = centers[1]-centers[0]
    return centers - h/2.

# Returns the domain of a binning given the centers
def bounding_box_from_centers(centers):
    h = centers[1]-centers[0]
    xmin = centers[0]-h/2.
    xmax = centers[-1]+h/2.
    return sp.array([xmin,xmax])

# Defines a dot product with my normalization
def dot(v1,v2,h=1.0):
    v1r = v1.ravel()
    v2r = v2.ravel()
    G = len(v1)
    assert( len(v2)==G )
    return sp.sum(v1r*v2r*h/(1.*G))

# Comptues a norm with my normalization
def norm(v,h=1.0):
    v_cc = np.conj(v)
    return sp.sqrt(dot(v,v_cc,h))

# Normalizes vectors (stored as columns of a 2D numpy array)
def normalize(vectors, grid_spacing=1.0):
    """ Normalizes vectors stored as columns of a 2D numpy array """
    K = vectors.shape[1] # number of vectors
    G = vectors.shape[0] # length of each vector

    # Set volume element h. This takes a little consideration
    if (isinstance(grid_spacing,NUMBER)):
        h = grid_spacing
    elif (isinstance(grid_spacing,ARRAY)):
        grid_spacing = sp.array(grid_spacing)
        h = sp.prod(grid_spacing)
    else:
        print 'ERROR: what kind of thing is grid_spacing?'
        print type(grid_spacing)
        raise

    assert (h > 0)

    norm_vectors = sp.zeros([G,K])
    for i in range(K):

        # Extract v from list of vectors
        v = vectors[:,i]

        # Flip sign of v so that last element is nonnegative
        if (v[-1] < 0):
            v = -v

        # Normalize v and save in norm_vectors
        norm_vectors[:,i] = v/norm(v)

    # Return array with normalized vectors along the columns
    return norm_vectors

# Construct an orthonormal basis of order alpha from 1d legendre polynomials
def legendre_basis_1d(G, alpha, grid_spacing=1.0):

    # Create grid of centred x-values ranging from -1 to 1
    x_grid = (sp.arange(G) - (G-1)/2.)/(G/2.)

    # First create an orthogonal (not necessarily normalized) basis
    raw_basis = sp.zeros([G,alpha])
    for i in range(alpha):
        c = sp.zeros(alpha)
        c[i] = 1.0
        raw_basis[:,i] = legval(x_grid,c)

    # Normalize basis
    basis = normalize(raw_basis, grid_spacing)

    # Return normalized basis
    return basis

# Construct an orthonormal basis of order alpha from 2d legendre polynomials
def legendre_basis_2d(Gx, Gy, alpha, grid_spacing=[1.0,1.0]):

    # Compute x-coords and y-coords, each ranging from -1 to 1
    x_grid = (sp.arange(Gx) - (Gx-1)/2.)/(Gx/2.)
    y_grid = (sp.arange(Gy) - (Gy-1)/2.)/(Gy/2.)

    # Create meshgrid of these
    xs,ys = np.meshgrid(x_grid,y_grid)

    basis_dim = alpha*(alpha+1)/2
    G = Gx*Gy
    raw_basis = sp.zeros([G,basis_dim])
    k = 0
    for a in range(alpha):
        for b in range(alpha):
            if a+b < alpha:
                c = sp.zeros([alpha,alpha])
                c[a,b] = 1
                raw_basis[:,k] = \
                    legval2d(xs,ys,c).T.reshape([G])
                k += 1

    # Normalize this basis using my convension
    basis = normalize(raw_basis, grid_spacing)
    
    # Return normalized basis
    return basis