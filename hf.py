"""
this file implements the hessian free optimization routine for common functions, the user should only give a function and its grad
"""



from .util import *
from .cg import *

"""
implements the hessian-free optimization routine, which is used for some future experiments

@param: func, a lambda function with input as input_shape, output real scalar
@param: grad, a lambda function with input as input_shape, output the same shape as input
@param: input_shape, a shape form that should comply with the std of numpy

@param: lamb_updator, a lambda take into one the original lambda, the other the reduction ratio as an observation. If None, never update lambda
"""
def hf_opt(func, grad, input_shape, max_iter=1000, lamb_updator=None, lamb_0=0.001, epi=0.00005):
    # first by given the target objective, we construct two-order sub-objective
    gamma = 1; # gamma controls the initial values' stochasity
    theta_i = gamma* np.random.randn(*input_shape);
    lamb_i = lamb_0; # the damping value, which could be adjusted with a TRUST_REGION criterion, this value is quite important
    former_val = func(theta_i);

    if(lamb_updator==None):
        lamb_updator=lambda lamb, r_r: lamb; # a trivial updator, do nothing


    iter= 0;
    # begin the optimization loop
    while(iter<max_iter):
        # print some debug info
        if(DEBUG):
            print("HF Iter %d Loss %f" % (iter, former_val));

        # begin the critical part
        cur_grad = grad(theta_i);
        custom_MAT_VEC_MULT  = lambda d_i: H_approx(grad, theta_i, d_i)+lamb_i*d_i;
        # first construct the approximation
        quad_approx = Quad_Func(A=None, b= cur_grad, c=former_val, MAT_VEC_MULT = custom_MAT_VEC_MULT);

        p_i, min_val = cg_optimize(quad_approx, epi); # min_val is just q(p_i)
        # update the theta
        theta_i = theta_i + p_i;

        cur_val = func(theta_i);
        
    
        # begin to update the lambda_i by computing the so-called "reduction ratio"
        reduction_ratio = (cur_val - former_val)/(min_val - former_val);
        
        lamb_i = lamb_updator(lamb_i, reduction_ratio);

        # update the former value
        former_val = cur_val;
        iter+=1;
    return theta_i;


"""
@param: grad, as explained above, while the d is the direction along which to evaluate the hessian
@param: theta_i, the func's current prober point
@param: d_i, which is an interface for the conjugate gradient opt-er

Use such a petit-routine to approximate the curvature
"""


def H_approx(grad, theta_i, d_i):
    # I still have no idea the infiniestimal step should be ? And it's relation to the stabilit.
    episilon = 0.001; # this could only be determined empirically
    return (grad(theta_i+episilon*d_i)-grad(theta_i))/episilon;



"""
Implements the lamb updating technique
@param: lamb, the current value
@param: r_r, reduction ratio
following the paper's suggestions
"""

def trust_region_adjust(lamb, r_r):
    if(r_r<0.25):
        lamb = 1.5*lamb;
    elif(r_r>0.75):
        lamb = 0.6777*lamb;
    else:
        lamb;
    return lamb;


if(__name__=='__main__'):
    n = 1000;
    A = np.random.randn(n,n);
    b = np.random.randn(n,1);

    # FAKE_MAT_VEC_MULT = lambda x: (np.random.randn(n,1));

    # we can always be assured a random matrix can be invertialble a.e
    quad = Quad_Func(A+A.T, b ,0);

    x = np.random.randn(n,1);
    print(['func_val:', quad.func(x)]);
    print(['grad:', quad.grad(x)]);

    print("True Minimum: %f" % (-0.5*MULT(b.T, MULT(np.linalg.inv(A+A.T), b))));

    hf_opt(quad.func, quad.grad, input_shape=[n,1], lamb_updator=trust_region_adjust, lamb_0=0.0);

    pass;









