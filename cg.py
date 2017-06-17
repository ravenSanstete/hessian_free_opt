# implemenation of the conjugate gradient optimization method

# import numpy as np


from .util import *



"""
This defines the quadratic family, which is an elementary class and the objective of Cg algorithm

self.A: the two-order matrix, n*n, no need to check the symmetry of such a matrix
self.b: the one-order coeff n*1
self.c: the bias, 1

input x: n*1
"""

# DEBUG = True;

# # use some global macro
# MUL=np.matmul;


class Quad_Func:
    def __init__(self, A, b, c, MAT_VEC_MULT=None):
        if(MAT_VEC_MULT==None):
            assert(A.shape[0]==A.shape[1]);
            assert((A==A.T).all);
            self.MAT_VEC_MULT = lambda d: MULT(A, d); # to avoid the duplicate semantics
        else:
            self.MAT_VEC_MULT = MAT_VEC_MULT;

        self.func = lambda x: (0.5*MULT(x.T, self.MAT_VEC_MULT(x)) + MULT(b.T, x) + c);
        self.grad = lambda x: (self.MAT_VEC_MULT(x) + b);

        # other not quite important parameters
        self.b = b;
        self.c = c;
        self.dim = b.shape[0];



"""
this function do a conjugate gradient optimization for the quad func

@param: quad, an instance of the quad function
@param: epi, which is the stop criterion, 
|| Deprecated @param: max_iter, as the name indicates (deprecated)

@param: MAT_VEC_MULT: which should be a lambda, which takes in  a vector(the direction), a user-provided routine for a more efficient matrix-vector multiplication, especially when the two-order matrix is a Hessian

@return: x_i, the last prober
@return: func(x_i), the approximated minimum value


epi is for the relative per-iteration progress
"""




def cg_optimize(quad, epi=0.001):
    ## first generate a random prober point
    gamma = 1.0; # do some precondition on the initial prober point
    k = 10; # we compute the 10-iteration average progress 
    
    x_i = gamma*np.random.randn(quad.dim, 1);
    d_i = -quad.grad(x_i);
    k_former_func_vals = np.zeros(k); ## like the replay memory


    iter=0;
    while(True):
        # compute or update the relative per-iteration progress here
        cur_val = quad.func(x_i);
        k_former_val = k_former_func_vals[np.mod(iter,k)];
        if(iter>=k and cur_val<k_former_val):
            prog_per_iter = np.abs((cur_val - k_former_val)/k_former_val)/k;
            if(prog_per_iter<epi):
                break; # which means the progress is trivial
            
        k_former_func_vals[np.mod(iter, k)] = quad.func(x_i);
        
            
        # do line search
        alpha = -(MULT(d_i.T, quad.grad(x_i)))/(MULT(d_i.T, quad.MAT_VEC_MULT(d_i)));
        # update x_i
        x_i = x_i + alpha*d_i;
        # this is stored for a future reuse
        new_grad = quad.grad(x_i);
        # update d_i
        beta = MULT(new_grad.T, quad.MAT_VEC_MULT(d_i))/MULT(d_i.T, quad.MAT_VEC_MULT(d_i));
        d_i = -new_grad + beta*d_i;

        if(DEBUG and False):
            print("CG Iter: %d Loss: %f" % (iter, cur_val));
            
        
        iter+=1; # update the iteration value
    return x_i, quad.func(x_i);

        
    

    
            



if(__name__=="__main__"):
    n = 500;
    A = np.random.randn(n,n);
    b = np.random.randn(n,1);

    # FAKE_MAT_VEC_MULT = lambda x: (np.random.randn(n,1));

    # we can always be assured a random matrix can be invertialble a.e
    quad = Quad_Func(A+A.T, b ,0);
    
    x = np.random.randn(n,1);
    print(['func_val:', quad.func(x)]);
    print(['grad:', quad.grad(x)]);

    cg_optimize(quad);
    print("True Minimum: %f" % (-0.5*MULT(b.T, MULT(np.linalg.inv(A+A.T), b))));


