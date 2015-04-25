import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import det, eigh, solve, eigvalsh, inv
import scipy.optimize as opt
import laplacian

import utils
import laplacian
import maxent
import time

# Put hard bounds on how big or small t can be
# T_MIN especially seems to help convergence
T_MAX = 100
T_MIN = -100
LOG_E_RANGE = 100
PHI_MAX = 50
PHI_MIN = -50

class Results(): pass;

# Represents a point along the MAP curve
class MAP_curve_point:

    def __init__(self, t, Q, log_E, details=False):
        self.t = t
        self.Q = Q
        self.phi = utils.prob_to_field(Q)
        self.log_E = log_E
        #self.details = details

# Represents the MAP curve
class MAP_curve:

    def __init__(self):
        self.points = []
        self._is_sorted = False

    def add_point(self, t, Q, log_E, details=False):
        #print 'Added point at t==%f'%t
        point = MAP_curve_point(t, Q, log_E, details)
        self.points.append(point)
        self._is_sorted = False

    def sort(self):
        self.points.sort(key=lambda x: x.t)
        self._is_sorted = True

    # Use this to get actual points along the MAP curve.
    # Ensures that points are sorted
    def get_points(self):
        if not self._is_sorted:
            self.sort()
        return self.points

    def get_maxent_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[0]
        assert(p.t == -sp.Inf)
        return p

    def get_histogram_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[-1]
        assert(p.t == sp.Inf)
        return p

    def get_log_evidence_ratios(self, finite=True):
        log_Es = sp.array([p.log_E for p in self.points])
        ts = sp.array([p.t for p in self.points])

        if finite:
            indices = (log_Es > -np.Inf)*(ts > -np.Inf)*(ts < np.Inf)
            return log_Es[indices], ts[indices]
        else:
            return log_Es, ts


# Convention: action, gradient, and hessian are G/N * the actual.
# This provides for more robust numerics

# Evaluate the action of a field given smoothness criteria
def action(phi, R, Delta, t, phi_in_kernel=False):
    G = 1.*len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    ones_col = sp.mat(sp.ones(G)).T

    if phi_in_kernel:
        S_mat = G*R_col.T*phi_col + G*ones_col.T*quasiQ_col
    else:
        S_mat = 0.5*sp.exp(-t)*phi_col.T*Delta_sparse*phi_col \
           + G*R_col.T*phi_col \
           + G*ones_col.T*quasiQ_col

    S = S_mat[0,0]
    assert np.isreal(S)
    return S

# Evaluate action gradient w.r.t. a field given smoothness criteria
def gradient(phi, R, Delta, t):
    G = 1.*len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    grad_col = sp.exp(-t)*Delta_sparse*phi_col + G*R_col - G*quasiQ_col
    grad = sp.array(grad_col).ravel()
    assert all(np.isreal(grad))
    return grad

# Evaluate action hessain w.r.t. a field given smoothness criteria
# NOTE: returns sparse matrix, not dense matrix!
def hessian(phi, R, Delta, t):
    G = 1.*len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    Delta_sparse = Delta.get_sparse_matrix()
    return sp.exp(-t)*Delta_sparse + G*diags(quasiQ,0)

# Get log ptgd of the maxent density
def log_ptgd_at_maxent(N, phi_M, R, Delta):
    kernel_dim = Delta._kernel_dim
    M = utils.field_to_prob(phi_M)
    M_on_kernel = sp.zeros([kernel_dim, kernel_dim])
    kernel_basis = Delta._kernel_basis
    lambdas = Delta._eigenvalues
    for a in range(int(kernel_dim)):
        for b in range(int(kernel_dim)):
            psi_a = sp.ravel(kernel_basis[:,a])
            psi_b = sp.ravel(kernel_basis[:,b])
            M_on_kernel[a,b] = sp.sum(psi_a*psi_b*M)

    # Compute log occam factor at infinity
    log_Occam_at_infty = -0.5*sp.log(det(M_on_kernel)) \
                         - 0.5*sp.sum(sp.log(lambdas[kernel_dim:]))

    assert np.isreal(log_Occam_at_infty)

    # Compute the log likelihod at infinty
    log_likelihood_at_infty = -N*sp.sum(phi_M*R) - N

    assert np.isreal(log_likelihood_at_infty)

    # Compute the log posterior (not sure this is right)
    log_ptgd_at_maxent = log_likelihood_at_infty + log_Occam_at_infty

    assert np.isreal(log_ptgd_at_maxent)
    return log_ptgd_at_maxent

# Computes the log of ptgd at t
def log_ptgd(N, phi, R, Delta, t):
    G = 1.*len(phi)
    alpha = 1.*Delta._alpha
    kernel_dim = 1.*Delta._kernel_dim
    H = hessian(phi, R, Delta, t)
    H_prime = H.todense()*sp.exp(t)

    S = action(phi, R, Delta, t)   
    assert np.isreal(S)

    # First try computing log determinant straight away
    log_det = sp.log(det(H_prime))

    # If result is not finite and real, try computing the sum of eigenvalues,
    # forcing the eigenvalues to be real and nonnegative
    if not (np.isreal(log_det) and np.isfinite(log_det)):
        #print 'Warning: log_det becoming difficult to compute.'
        lambdas = abs(eigvalsh(H_prime))
        log_det = sp.sum(sp.log(lambdas))

    assert (np.isreal(log_det) and np.isfinite(log_det))
    
    # Compute contribution from finite t
    log_ptgd = -(N/G)*S + 0.5*kernel_dim*t - 0.5*log_det

    details = Results()
    details.S = S
    details.N = N
    details.G = G
    details.kernel_dim = kernel_dim
    details.t = t
    details.log_det = log_det
    details.phi = phi

    return log_ptgd, details

# Computes error bars on phi at t
def get_dQ_sq(N, phi, R, Delta, t):
    G = 1.*len(phi)
    Q = utils.field_to_prob(phi)

    # If t is finite, just compute diagonal of covariance matrix
    if np.isfinite(t):

        H = (N/G)*hessian(phi, R, Delta, t)

        dQ_sq = np.zeros(int(G))
        for i in range(int(G)):
            delta_vec = np.zeros(int(G))
            delta_vec[i] = 1.0
            v = Q - delta_vec
            a = spsolve(H, v)
            dQ_sq[i] = (Q[i]**2)*np.sum(v*a)

    # If t is not finite, this is a little more sophisticated 
    # but not harder computationally
    else:
        H = (N/G)*spdiags(np.exp(-phi),0,G,G)
        psis = np.mat(Delta._kernel_basis)
        H_tilde = psis.T * H * psis
        H_tilde_inv = inv(H_tilde)
        #dphi_cov = psis*H_tilde_inv*psis.T
        
        dQ_sq = np.zeros(int(G))
        for i in range(int(G)):
            delta_vec = np.zeros(int(G))
            delta_vec[i] = 1.0
            v_col = sp.mat(Q - delta_vec).T
            v_proj = psis.T * v_col
            dQ_sq[i] = (Q[i]**2)*(v_proj.T*H_tilde_inv*v_proj)[0,0]

    # Note: my strange normalization conventions might be causing problmes
    # Might be missing factor of G in here
    return dQ_sq



# Computes predictor step
def compute_predictor_step(phi, R, Delta, t, direction, resolution):

    # Make sure direction is just a sign
    assert(direction==1 or direction==-1)

    # Get current probability dist
    Q = utils.field_to_prob(phi)

    G = 1.*len(Q)

    # Get hessian
    H = hessian(phi, R, Delta, t)

    # Comput rho, which indicates direction of step
    rho = G*spsolve(H, Q - R )
    assert all(np.isreal(rho))

    denom = sp.sqrt(sp.sum(rho*Q*rho))
    assert np.isreal(denom)
    assert denom > 0

    # Compute dt based on value of epsilon (the resolution)
    dt = direction*resolution/denom

    # Return phi_new and new t_new
    phi_new = phi + rho*dt
    t_new = t + dt
    return phi_new, t_new

# Computes corrector step
def compute_corrector_step(phi, R, Delta, t, tollerance=1E-5, report_num_steps=False):

    # Evaluate the probabiltiy distribution
    Q = utils.field_to_prob(phi)

    # Evaluate action
    S = action(phi, R, Delta, t)

    # Perform corrector steps until until phi converges
    num_corrector_steps = 0
    num_backtracks = 0
    while True:

        # Compute the gradient
        v = gradient(phi, R, Delta, t)
        
        # Compute the hessian
        H = hessian(phi, R, Delta, t)

        # Solve linear equation to get change in field
        try:
            dphi = -spsolve(H,v)
            assert all(np.isreal(dphi))
            assert all(np.isfinite(dphi))        
        except:
            break

        # Compute corresponding change in action
        dS = sp.sum(dphi*v)
        
        # Reduce step size until in linear regime
        beta = 1.0
        while True:

            # Compute new phi 
            phi_new = phi + beta*dphi
            try:
                Q_new = utils.field_to_prob(phi_new, regularize=True)
            except:
                print 'Error: setting Q_new failed.'
                print 'phi == %s'%str(phi)
                print 'dphi == %s'%str(dphi)
                print 'beta== %s'%str(beta)
                raise

            # Make sure beta isn't fucking up
            if beta < 1E-10:
                print ' --- Something is wrong. ---'
                print 'beta == %f'%beta
                print 'dS == %f'%dS
                print 'S == %f'%S
                print 'S_new == %f'%S_new
                print 'phi == ' + str(phi)
                print 'dphi == ' + str(dphi)
                print 'v == ' + str(v)
                print 'geo_dist == %f'%utils.geo_dist(Q_new,Q)
                print ''
                time.sleep(1)

            # If geodesic distance is less than tollerance, continue
            gd = utils.geo_dist(Q_new, Q)
            if gd < tollerance:
                break;

            # Compute new action
            S_new = action(phi_new, R, Delta, t)

            # Check for linear regime 
            if (S_new <= S + 0.5*beta*dS):
                break

            # If not in linear regime backtrack value of beta
            else:
                num_backtracks+=1
                beta *= 0.5    

        # Break out of loop if Q_new is close enough to Q
        if gd < tollerance:
            break
        
        # Break out of loop with warning if S_new > S. Should not happen,
        # but not fatal if it does. Just means less precision
        elif S_new-S > 0:
            print 'Warning: S_change > 0. Terminating corrector steps.'
            break

        # Otherwise, continue with corrector step
        else:
            # New phi, Q, and S values have already been computed
            phi = phi_new
            Q = Q_new
            S = S_new
            num_corrector_steps += 1

    # After corrector loop has finished, return field
    # Also return stepping stats if requested
    if report_num_steps:
        return phi, num_corrector_steps, num_backtracks
    else:
        return phi

# The core algorithm of DEFT, used for both 1D and 2D density esitmation
def compute_map_curve(N, R, Delta, resolution=1E-2, tollerance=1E-3):
    """ Traces the map curve in both directions

    Args:

        R (numpy.narray): 
            The data histogram

        Delta (Smoothness_operator instance): 
            Effectiely defines smoothness

        resolution (float): 
            Specifies max distance between neighboring points on the 
            MAP curve

    Returns:

        map_curve (list): A list of MAP_curve_points

    """

    #resolution=3.14E-2
    #tollerance=1E-3

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    alpha = Delta._alpha
    kernel_basis = Delta.get_kernel_basis()
    kernel_dim = Delta.get_kernel_dim()

    # Make sure the smoothness_operator has the right shape
    assert(G == len(R))

    # Make sure histogram is nonnegative
    assert(all(R >= 0))

    # Make sure that enough elements of counts_array contain data
    assert(sum(R >= 0) > kernel_dim)

    # Inialize map curve
    map_curve = MAP_curve()

    #
    # First compute histogram stuff
    #

    # Get normalied histogram and correpsonding field
    R = R/sum(R)
    phi_0 = utils.prob_to_field(R)
    log_E_R = -np.Inf
    t_R = np.Inf
    map_curve.add_point(t_R, R, log_E_R)

    #
    # Now compute maxent stuff
    #

    # Compute the maxent field and density
    phi_infty, success = maxent.compute_maxent_field(R, kernel_basis)

    # Convert maxent field to probability distribution
    M = utils.field_to_prob(phi_infty)

    # Compute the maxent log_ptgd
    # Important to keep this around to compute log_E at finite t
    log_ptgd_M = log_ptgd_at_maxent(N, phi_infty, R, Delta)

    # This corresponds to a log_E of zero
    log_E_M = 0
    t_M = -sp.Inf
    map_curve.add_point(t_M, M, log_E_M)

    # Set maximum log evidence ratio so far encountered
    log_E_max = -np.Inf #0

    # Smoothness parameter t. 
    # t_start corresponds to ell_0 = sqrt(G)
    t_start = sp.log(N) - alpha*sp.log(G) 
    #print 'Starting at t == %f'%t_start

    # Compute phi_start by executing a corrector step starting at maxent dist
    phi_start = compute_corrector_step( phi_infty, R, Delta, t_start, \
                                        tollerance=tollerance, \
                                        report_num_steps=False)
    
    # Convert starting field to probability distribution
    Q_start = utils.field_to_prob(phi_start)

    # Compute log ptgd
    log_ptgd_start, start_details = log_ptgd(N, phi_start, R, Delta, t_start)

    # Compute corresponding evidence ratio
    log_E_start = log_ptgd_start - log_ptgd_M

    # Adjust max log evidence ratio
    log_E_max = log_E_start if (log_E_start > log_E_max) else log_E_max

    # Set start as first map curve point 
    map_curve.add_point(t_start, Q_start, log_E_start) #, start_details)

    # Trace map curve in both directions
    for direction in [-1,+1]:
        
        # Start iteration from central point
        phi = phi_start
        t = t_start
        Q = Q_start
        log_E = log_E_start

        if direction == -1:
            Q_end = M
        else:
            Q_end = R

        # Keep stepping in direction until read the specified endpoint
        while True:

            # Test distance to endpoint
            if utils.geo_dist(Q_end,Q) <= resolution:
                break

            # Take predictor step 
            phi_pre, t_new = compute_predictor_step( phi, R, Delta, t, \
                                                     direction=direction, \
                                                     resolution=resolution )

            # Compute new distribution
            Q_pre = utils.field_to_prob(phi_pre)

            #print 'geo_dist(Q_pre,Q) == %f'%utils.geo_dist(Q_pre,Q)

            # Perform corrector stepsf to get new phi
            phi_new = compute_corrector_step( phi_pre, R, Delta, t_new, \
                                              tollerance=tollerance, \
                                              report_num_steps=False)

            # Compute new distribution
            Q_new = utils.field_to_prob(phi_new)

            # Compute log ptgd
            log_ptgd_new, details_new = log_ptgd(N, phi_new, R, Delta, t_new)

            # Compute corresponding evidence ratio
            log_E_new = log_ptgd_new - log_ptgd_M

            # Take step
            t = t_new
            Q = Q_new   
            phi = phi_new
            log_E = log_E_new
            details = details_new

            # Add new point to map curve
            map_curve.add_point(t, Q, log_E) #, details_new)

            # Adjust max log evidence ratio
            log_E_max = log_E if (log_E > log_E_max) else log_E_max

            # Terminate if log_E is too small. But don't count
            # the t=-inf endpoint when computing log_E_max 
            if (log_E_new < log_E_max - LOG_E_RANGE):
                #print 'Log_E too small. Exiting at t == %f'%t
                break

            #print '\ngeo_dist(Q_new,Q) == %f'%utils.geo_dist(Q_new,Q)  

            # Terminate if t is too large or too small
            if t > T_MAX:
                #print 'Warning: t = %f is too positive. Stopping trace.'%t 
                break
            elif t < T_MIN:
                #print 'Warning: t = %f is too negative. Stopping trace.'%t 
                break

    # Sort points along the MAP curve
    map_curve.sort()
    map_curve.t_start = t_start

    # Return the MAP curve to the user
    return map_curve

### Core DEFT algorithm ###

# The core algorithm of DEFT, used for both 1D and 2D density esitmation
def run(counts_array, Delta, resolution=3.14E-2, tollerance=1E-3, \
    details=False, errorbars=True, num_samples=20 ):
    """
    The core algorithm of DEFT, used for both 1D and 2D density estmation.

    Args:
        counts_array (numpy.ndarray): 
            A scipy array of counts. All counts must be nonnegative.

        Delta (Smoothness_operator instance): 
            An operator providing the definition of 'smoothness' used by DEFT
    """

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    kernel_dim = Delta.get_kernel_dim()

    # Make sure the smoothness_operator has the right shape
    assert(G == len(counts_array))

    # Make sure histogram is nonnegative
    assert(all(counts_array >= 0))

    # Make sure that enough elements of counts_array contain data
    assert(sum(counts_array >= 0) > kernel_dim)

    # Get number of data points and normalized histogram
    N = sum(counts_array)

    # Get normalied histogram
    R = 1.0*counts_array/N
    
    # Compute the MAP curve
    map_curve = compute_map_curve( N, R, Delta, \
                                   resolution=resolution, \
                                   tollerance=tollerance)

    # Identify the optimal density estimate
    points = map_curve.points
    log_Es = sp.array([p.log_E for p in points])
    log_E_max = log_Es.max()
    ibest = log_Es.argmax()
    star = points[ibest]
    Q_star = star.Q
    t_star = star.t
    phi_star = star.phi

    # Compute errorbars if requested
    if errorbars:   

        # Get list of map_curve points that with evidence ratio 
        # of at least 1% the maximum
        log_E_threshold = log_E_max + np.log(0.001)

        # Get points that satisfy threshold
        points_at = [p for p in points if p.log_E > log_E_threshold]

        #print '\n'.join(['%f\t%f'%(p.t,p.log_E) for p in points])

        # Get weights at each ell
        log_Es_at = np.array([p.log_E for p in points_at])
        log_Es_at -= log_Es_at.max()
        weight_ell = np.mat(np.exp(log_Es_at))
        
        # Get systematic variance due to changes in Q_ell at each ell
        dQ_sq_sys_ell = np.mat([(p.Q-Q_star)**2 for p in points_at])

        # Get random variance about Q_ell at each ell
        dQ_sq_rand_ell = np.mat([get_dQ_sq(N, p.phi, R, Delta, p.t) 
            for p in points_at])

        #print dQ_sq_rand_ell

        # Sum systematic and random components to variance
        dQ_sq_ell = dQ_sq_sys_ell + dQ_sq_rand_ell

        #print weight_ell.shape
        #print dQ_sq_ell.shape

        # Compute weighted averaged to get final dQ_sq
        dQ_sq_mat = weight_ell*dQ_sq_ell/sp.sum(sp.array(weight_ell))
        
        # Convert from matrix to array
        dQ_sq = sp.array(dQ_sq_mat).ravel()
        
        try:
            assert(all(np.isfinite(dQ_sq)))
        except:
            print [p.log_E for p in points_at]
            print weight_ell
            print dQ_sq_sys_ell
            print dQ_sq_rand_ell
            raise

        # Compute interval
        Q_ub = Q_star + np.sqrt(dQ_sq)
        Q_lb = Q_star - np.sqrt(dQ_sq)

    # Sample plausible densities from the posterior
    Q_samples = sp.zeros([0,0])
    if num_samples > 0:

        #print 't_star == ' + str(t_star)

        # Get list of map_curve points that with evidence ratio 
        # of at least 1% the maximum
        log_E_threshold = log_E_max + np.log(0.001)

        # Get points that satisfy threshold
        points_at = [p for p in points if p.log_E > log_E_threshold]

        # Get weights at each ell
        weights = np.array([np.exp(p.log_E) for p in points_at])

        # Compute eigenvectors of the Hessian
        # If t is finite, this is straight-forward
        if t_star > -np.Inf:
            h_star = hessian(phi_star, R, Delta, t_star)
            lambdas_unordered, psis_unordered = eigh(h_star.todense())
            ordered_indices = np.argsort(lambdas_unordered)
            psis = psis_unordered[:,ordered_indices]

        # If t is infinite but kernel is non-degenerate
        elif Delta._kernel_dim == 1:
            psis = Delta._eigenbasis

        # If t is infinite and kernel is degenerate and needs to be
        # diagonalized with respect to diag(Q_star)
        else:
            psis_ker = Delta._kernel_basis
            kd = Delta._kernel_dim
            h_ker = sp.zeros([kd, kd])
            psis = sp.zeros([G,G])
            for i in range(kd):
                for j in range(kd):
                    psi_i = sp.array(psis_ker[:,i])
                    psi_j = sp.array(psis_ker[:,j])
                    h_ker[i,j] = sp.sum(np.conj(psi_i)*psi_j*Q_star)
            _ , cs = eigh(h_ker)
            rhos = sp.mat(cs).T*psis_ker.T
            psis[:,:kd] = rhos.T
            psis[:,kd:] = Delta._eigenbasis[:,kd:]
            
        # Figure out how many samples to draw for each ell value
        candidate_ell_indices = range(len(points_at))
        candidate_ell_probs = weights/sp.sum(weights)
        ell_indices = np.random.choice(candidate_ell_indices, 
            size=num_samples, p=candidate_ell_probs)
        unique_ell_indices, ell_index_counts = np.unique(ell_indices, return_counts=True)

        # Draw samples at each lenghtscale
        Q_samples = sp.zeros([G,num_samples])
        num_samples_obtained = 0
        for k in range(len(unique_ell_indices)):
            ell_index = unique_ell_indices[k]
            num_samples_at_ell = ell_index_counts[k]

            p = points_at[ell_index]

            # If t is finite, figure out how many psis to use
            if p.t > -np.Inf:
                # Get hessian
                #H = (1.*N/G)*hessian(p.phi, R, Delta, p.t)
                H = hessian(p.phi, R, Delta, p.t)

                # Compute inverse variances below threshold
                inv_vars = []
                for i in range(G):
                    psi = psis[:,i]
                    psi_col = sp.mat(psi[:,None])
                    inv_var = (np.conj(psi_col.T)*H*psi_col)[0,0]
                    if i==0:
                        inv_vars.append(inv_var)
                    elif inv_var < (1.0E10)*min(inv_vars):
                        inv_vars.append(inv_var)
                    else:
                        break;
                assert all(np.isreal(inv_vars))
                psis_use = psis[:,:len(inv_vars)]

            # If t is finite, only use psis in kernel
            else:
                #H = 1.*N*spdiags(p.Q,0,G,G)
                H = 1.*G*spdiags(p.Q,0,G,G)
                kd = Delta._kernel_dim
                psis_use = psis[:,:kd]
                inv_vars = sp.zeros(kd)
                for i in range(kd):
                    psi_i = sp.mat(psis_use[:,i]).T
                    inv_var = (np.conj(psi_i.T)*H*psi_i)[0,0]
                    assert np.isreal(inv_var)
                    inv_vars[i] = inv_var
                
            # Make sure all inverse variances are greater than zero
            assert all(np.array(inv_vars) > 0)

            # Now draw samples at this ell!
            psis_use_mat = sp.mat(sp.array(psis_use))
            inv_vars = sp.array(inv_vars)
            num_psis_use = psis_use_mat.shape[1]
            
            # Perform initial sampling at this ell
            # Sample 10x more phis than needed if doing posterior pruning
            #M = 10*num_samples_at_ell
            M = num_samples_at_ell
            phi_samps = sp.zeros([G,M])
            sample_actions = sp.zeros(M)
            for m in range(M):
                
                # Draw random numbers for dphi coefficients
                r = sp.randn(num_psis_use)

                # Compute action used for sampling
                S_samp = np.sum(r**2)/2.0 # Action for specific sample

                # Construct sampled phi
                sigmas = 1./np.sqrt((1.*N/G)*inv_vars)
                a = sp.mat(r*sigmas)
                dphi = sp.array(a*psis_use_mat.T).ravel()
                phi = p.phi + dphi
                phi_samps[:,m] = phi
                
                # Compute true action for phi_samp
                phi_in_kernel = (p.t == -np.Inf)

                # USE THIS IF YOU DON'T WANT TO DO POSTERIOR PRUNING
                # RIGHT NOW I DON'T THINK THIS SHOULD BE DONE
                # THIS LACK OR PRUNING CREATES FLIPPY TAILS ON THE POSTERIOR
                # SAMPLES, BUT THIS GENUINELY REFLECTS THE HESSIAN I THINK
                if True:
                    sample_actions[m] = 0
                else:
                    S = (1.*N/G)*action(phi, R, Delta, p.t, phi_in_kernel=phi_in_kernel)
                    sample_actions[m] = -S+S_samp

            # Now compute weights. Have to make bring actions into a 
            # sensible range first
            sample_actions -= sample_actions.min()
            

            # Note: sometimes all samples except one have nonzero weight
            # The TINY_FLOAT32 here regularizes these weights so that
            # the inability to sample well doesn't crash the program
            sample_weights = sp.exp(-sample_actions) + utils.TINY_FLOAT32

            # Choose a set of samples. Do WITHOUT replacement.
            try:
                sample_probs = sample_weights/np.sum(sample_weights)
                sample_indices = sp.random.choice(M, size=num_samples_at_ell, replace=False, p=sample_probs)
            except:
                print sample_weights
                print sample_probs
                print num_samples_at_ell
                raise

            #print p.t
            #print sample_weights
            #print np.sort(sample_probs)[::-1]
            for n in range(num_samples_at_ell):
                index = sample_indices[n]
                phi = phi_samps[:,index]
                m = num_samples_obtained + n
                Q_samples[:,m] = utils.field_to_prob(phi, regularize=True)

            num_samples_obtained += num_samples_at_ell

        # Randomize order of samples
        indices = np.arange(Q_samples.shape[1])
        np.random.shuffle(indices)
        Q_samples = Q_samples[:,indices]


    # Return density estimate along with histogram on which it is based
    if details:
        return Q_star, R, Q_ub, Q_lb, Q_samples, map_curve
    else:
        return Q_star, R



