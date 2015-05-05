import scipy as sp
import numpy as np
from scipy.linalg import solve, det, norm

import utils
import laplacian

PHI_STD_REG = utils.PHI_STD_REG 

# Compute field from its coefficients in a basis
def coeffs_to_field(coeffs, kernel):
    """ For maxent algorithm. """

    # Get dimensions and check that they are valid
    kernel_dim = kernel.shape[1]
    G = kernel.shape[0]
    assert(len(coeffs)==kernel_dim)
    
    # Convert to matrices
    kernel_mat = sp.mat(kernel) # G x kernel_dim matrix
    coeffs_col = sp.mat(coeffs).T # kernel_dim x 1 matrix
    field_col = kernel_mat*coeffs_col # G x 1 matrix

    return sp.array(field_col).ravel() # Returns an array

# Compute the action of a field given its coefficients in a basis
def action_per_datum_from_coeffs(coeffs, R, kernel, phi0=False, 
    regularized=True):
    """ For optimizer. Computes action from coefficients. """
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    phi = coeffs_to_field(coeffs, kernel)
    quasiQ = utils.field_to_quasiprob(phi+phi0)
    current_term = sp.sum(R*phi)
    nonlinear_term = sp.sum(quasiQ)
    s = current_term + nonlinear_term
    if regularized:
        G = len(phi)
        s += (.5/G)*sum(phi**2)/(PHI_STD_REG**2)
    return s

# Compute the action gradient w.r.t field coefficients in a basis
def gradient_per_datum_from_coeffs(coeffs, R, kernel, phi0=False, 
    regularized=True):
    """ For optimizer. Computes gradient from coefficients. """
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    phi = coeffs_to_field(coeffs, kernel)
    quasiQ = utils.field_to_quasiprob(phi+phi0)
    G = len(phi)

    R_row = sp.mat(R) # 1 x G
    quasiQ_row = sp.mat(quasiQ) # 1 x G
    reg_row = (1./G)*sp.mat(phi)/(PHI_STD_REG**2) # 1 x G
    kernel_mat = sp.mat(kernel) # G x kernel_dim

    mu_R_row = R_row*kernel_mat # 1 x kernel_dim
    mu_quasiQ_row = quasiQ_row*kernel_mat # 1 x kernel_dim
    mu_reg_row = reg_row*kernel_mat

    if regularized:
        grad_row = mu_R_row - mu_quasiQ_row + mu_reg_row
    else:
        grad_row = mu_R_row - mu_quasiQ_row

    return sp.array(grad_row).ravel() # Returns an array

# Compute the action hessin w.r.t field coefficients in a basis
def hessian_per_datum_from_coeffs(coeffs, R, kernel, phi0=False, 
    regularized=True):
    """ For optimizer. Computes hessian from coefficients. """
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    phi = coeffs_to_field(coeffs, kernel)
    quasiQ = utils.field_to_quasiprob(phi+phi0)
    
    kernel_mat = sp.mat(kernel) # G x kernel_dim matrix
    H = sp.mat(sp.diag(quasiQ)) # G x G matrix

    if regularized:
        G = len(phi)
        H += (1./G)*sp.diag(np.ones(G))/(PHI_STD_REG**2)

    hessian_mat = kernel_mat.T*H*kernel_mat # kernel_dim^2 matrix

    return sp.array(hessian_mat) # Returns an array

# Computes the maximum entropy probaiblity distribution
def compute_maxent_prob_1d(R, kernel, h=1.0, report_num_steps=False, 
    phi0=False):
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    field, num_corrector_steps, num_backtracks = \
        compute_maxent_field(R, kernel, report_num_steps=True, phi0=phi0)
    Q = utils.field_to_prob(field+phi0)/h
    if report_num_steps:
        return Q, num_corrector_steps, num_backtracks
    else:
        return Q


# Computes the maximum entropy probaiblity distribution
def compute_maxent_prob_2d(R, kernel, grid_spacing=[1.0,1.0],\
        report_num_steps=False, phi0=False):
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    phi, num_corrector_steps, num_backtracks = \
        compute_maxent_field(R, kernel, report_num_steps=True)
    h = grid_spacing[0]*grid_spacing[1]
    Q = utils.field_to_prob(phi+phi0)/h
    if report_num_steps:
        return Q, num_corrector_steps, num_backtracks
    else:
        return Q

# Compute the maxent field 
def compute_maxent_field(R, kernel, report_num_steps=False, 
    geo_dist_tollerance=1.0E-3, phi0=False, grad_tollerance=1E-5):
    """
    Computes the maxent field from a histogram and kernel
    
    Args:
        R (numpy.narray): 
            Normalized histogram of the raw data. Should have size G
            
        kernel (numpy.ndarray): 
            Array of vectors spanning the smoothness operator kernel. Should
            have size kernel_dim x G
            
    Returns:
        
        phi: 
            The MaxEnt field. 
    """
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    # Get dimension of kernel
    kernel_dim = kernel.shape[1]

    # Make sure kernel vectors are same size as R
    G = len(R)
    assert(kernel.shape[0]==G)

    # set coefficients to zero
    if kernel_dim > 1:
        coeffs = sp.zeros(kernel_dim)
        #coeffs = sp.randn(kernel_dim)
    else:
        coeffs = sp.zeros(1)

    # Evaluate the probabiltiy distribution
    phi = coeffs_to_field(coeffs, kernel)
    
    phi = sp.array(phi).ravel()
    phi0 = sp.array(phi0).ravel()
    #print phi+phi0
    Q = utils.field_to_prob(phi+phi0)

    # Evaluate action
    s = action_per_datum_from_coeffs(coeffs, R, kernel, phi0)

    # Perform corrector steps until until phi converges
    num_corrector_steps = 0
    num_backtracks = 0
    while True:

        if kernel_dim == 1:
            success = True
            break

        # Compute the gradient
        v = gradient_per_datum_from_coeffs(coeffs, R, kernel, phi0)
        
        # If gradient is not detectable, we're already done!
        if norm(v) < G*utils.TINY_FLOAT32:
            break

        # Compute the hessian
        Lambda = hessian_per_datum_from_coeffs(coeffs, R, kernel, phi0)

        # Solve linear equation to get change in field
        # This is the conjugate gradient method
        da = -sp.real(solve(Lambda,v))

        # Compute corresponding change in action
        ds = sp.sum(da*v)

        # This should always be satisifed
        if (ds > 0):
            print 'Warning: ds > 0. Quitting compute_maxent_field.'
            break

        # Reduce step size until in linear regime
        beta = 1
        success = False
        while True:

            # Compute new phi and new action
            coeffs_new = coeffs + beta*da
            s_new = action_per_datum_from_coeffs(coeffs_new,R,kernel,phi0)

            # Check for linear regime
            if s_new <= s + 0.5*beta*ds:
                break

            # Check to see if bet is too small and algorithm is failing
            elif beta < 1E-20:
                print 'Error: compute_maxent_field not converging'
                raise    

            # If not in linear regime backtrack value of beta
            else:
                # pdb.set_trace()
                num_backtracks+=1
                beta *= 0.5

        # Compute new distribution
        phi_new = coeffs_to_field(coeffs_new, kernel)
        Q_new = utils.field_to_prob(phi_new+phi0)

        # Break out of loop if Q_new is close enough to Q
        if (utils.geo_dist(Q_new,Q) < geo_dist_tollerance) and (np.linalg.norm(v) < grad_tollerance):
            success = True
            break
        
        # Break out of loop with warning if S_new > S. Should not happen,
        # but not fatal if it does. Just means less precision
        elif s_new-s > 0:
            print 'Warning: action has increased. Terminating steps.'
            success = False
            break

        # Otherwise, continue with corrector step
        else:
            num_corrector_steps += 1

            # Set new coefficients.
            # New s, Q, and phi laready computed
            coeffs = coeffs_new
            s = s_new
            Q = Q_new
            phi = phi_new


    # Actually, should judge success by whether moments match
    if not success:
        print 'gradident norm == %f'%np.linalg.norm(v)
        print 'gradient tollerance == %f'%grad_tollerance
        print 'Failure! Trying Maxent again!'

    # After corrector loop has finished, return field
    # Also return stepping stats if requested
    if report_num_steps:
        return phi, num_corrector_steps, num_backtracks
    else:
        return phi, success

