import numpy as np
import findiff as fd
import importlib
from . import Frame_Builder as fb
importlib.reload(fb)
import os
import pickle
import tabulate
import matplotlib.pyplot as plt

def ssm_options():
    # option, keyword, default value, description
    opts = [
        ["Parameter", "keyword", "options", "default value"],
        ["Measure", "meas", "scaled, translated", "scaled"],
        ["Frame type", "fname", "see safari.framelist()", "custom"],
        ["Frame", "F", "N x L numpy array", "None"],
        ["Derivative of Frame", "dF", "N x L numpy array", "None"],
        ["Dual of Frame", "D", "N x L numpy array", "None"],
        ["Number of coefficients", "N", "positive integer", "50"],
        ["Length of frame", "L", "positive integer", "2**14"],
        ["Save path", "sav_path", "string (file path)", "None"]
    ]
    headers = opts[0]
    table = opts[1:]
    print(tabulate.tabulate(table, headers, tablefmt="grid"))

def eigen_value_decomp(A):
    try:
        eigenvalues, eigenvectors= np.linalg.eig(A)
        eigval_hat= np.linalg.inv(eigenvectors) @ A @ eigenvectors
        
        diagonal_MSE= np.linalg.norm( np.diag(np.diag(eigval_hat)) )  / np.linalg.norm( eigval_hat )           
        is_diag=False
        if diagonal_MSE>0.999: # why this value?
            is_diag=True
    except Exception:
        is_diag=False
        eigenvectors=None
        eigenvalues=None
    print("There are ", len(eigenvalues), "eigenvalues and the diagonalization is", is_diag, "\n")
    return eigenvalues, eigenvectors, is_diag

def get_effective_rank(A,tol=1e-12):
    """Compute the effective rank of a matrix."""
    U, s, Vh = np.linalg.svd(A)
    normalized_s = s / np.sum(s)
    normalized_s = normalized_s[normalized_s > tol]  # Filter out small singular values
    entropy = -np.sum(normalized_s * np.log(normalized_s))
    return np.exp(entropy)

class SSM:
    """
    Attributes: 
    Fobj, Frame object containing frame, dual, and derivative
    N, number of coefficients
    L, length of frame
    fname, name of the function used to generate frame (eg, legendre, fourier)
    meas, measure (eg, scaled, translated)
    """
    def __init__(self, **params):

        FOBJ_KEYS = {"fname", "F", "dF", "D", "N", "L", "dborder", "num_freqs", "redundancy", 
                     "range_min", "range_max", "fmin", "fmax", "rcond", "sigma_factor",
                     "freqs", "base_freq", "num_freq_levels", "freq_scale", "multiscale", "m"}
        self.fname = params.get("fname", 'custom') 
        self.meas = params.get("meas", 'scaled')
        save_path = params.get("sav_path", None)
        self.F = params.get("F", None)  
        self.dF = params.get("dF", None)
        self.D = params.get("D", None) 
        self.N = params.get("N", 32)

        fobj_params = {k: v for k, v in params.items() if k in FOBJ_KEYS}    
        
        # Establish frame to use
        if self.F is not None: # if the user has provided a frame, use it.
            print("Using provided frame.")
            self.Fobj = fb.Fobj(**fobj_params)

        else:  # else, build the frame using provided parameters
            if self.fname in ['legendre', 'fourier', 'chebyshev', 'laguerre', 'bernstein', 'gabor', 'daubechies']:
                self.Fobj = fb.Fobj(**fobj_params)
        
        # check whether there is a known closed-form solution for A, B
        if ((self.fname=='legendre' and self.meas=='scaled') or (self.fname=='legendre' and self.meas=='translated')): 
            #or (self.fname=='fourier' and self.meas=='translated')):
            self.hippo()
        else: # if not, default to numerical method
            self.safari()
        
        #Takes A, and diagonalize it. self.is_diag indicates whether the ssm can be *diagonalized*. (nothing about stability!!)
        # If self.id_diag is calculated to be False, then you should not use the diagonal solution.
        self.eig_val, self.eig_vec, self.B_diag, self.is_diag = self.diagonalize() 
           
        # Save if path is provided
        if save_path is not None:
            self.save(save_path)
            
    def save(self, path):
        """Save the current SSM object to the given path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"SSM object saved to {path}")

    @staticmethod
    def load(path):
        """Load an SSM object from the given path with exception handling."""
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            raise FileNotFoundError(f"No such file: '{path}'")
        except pickle.UnpicklingError:
            raise ValueError(f"The file at '{path}' is not a valid SSM pickle.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading: {e}")      

    def hippo(self):

        print("Using HiPPO closed-form solution for A, B")
        A = np.zeros((self.N,self.N))
        B = np.zeros((self.N,1))

        if self.fname=='legendre' and self.meas=='scaled':
            print('Generating HiPPO-LegS')
            for n in range(self.N):
                B[n] = np.sqrt(2*n+1)
                for k in range(n+1):
                    if n == k:
                        A[n,k] = n+1
                    else:
                        A[n,k] = np.sqrt(2*n+1)*np.sqrt(2*k+1) 

        elif self.fname=='legendre' and self.meas=='translated':
            print('Generating HiPPO-LegT')
            for n in range(self.N):
                B[n] = np.sqrt(2*n+1)
                for k in range(self.N):
                    if n <= k:
                        A[n,k] = np.sqrt(2*n+1)*np.sqrt(2*k+1)*((-1)**(n-k))
                    else:
                        A[n,k] = np.sqrt(2*n+1)*np.sqrt(2*k+1) 

        elif self.fname=='fourier' and self.meas=='translated':
            #self.N = self.N + 1 # add one for the DC term
            A = np.zeros((self.N,self.N))
            B = np.zeros((self.N,1))
            print('Generating HiPPO-FouT')
            for n in range(self.N):
                if n % 2 == 1:
                    B[n] = -np.sqrt(2)
                for k in range(self.N):
                    if n == 0 and k == 0:
                        A[n,k] = 1
                    if n%2 == 1: 
                        if k == 0: # n odd, k=0
                            A[n,k] = np.sqrt(2) 
                        elif k%2 == 1: # n,k both odd
                            A[n,k] = 2
                        elif k-n == 1: # n odd, k-n = 1
                            A[n,k] = -np.pi*(n+1)
                    elif k%2 == 1: 
                        if n == 0: # k odd, n=0
                            A[n,k] = np.sqrt(2)
                        elif n-k == 1: # k odd, n-k = 1
                            A[n,k] = np.pi*(k+1)
            B[0] = -2

        self.A = A
        self.B = B
    
    def safari(self): 

        self.N = self.Fobj.F.shape[0]
        L = self.Fobj.F.shape[1]

        if self.meas == 'scaled':
            dF = self.Fobj.dF 
            Theta = (np.arange(L)/L)*dF
            t = np.eye(self.N)

        elif self.meas == 'translated':
            Theta = self.Fobj.dF
            t = (self.Fobj.F[:,0][:,None] @ self.Fobj.D[:,0][None,:])*L

        A = t + (Theta @ self.Fobj.D.T)
        B = self.Fobj.F[:,L-1]
    
        self.A = A
        self.B = B[:,None]
    
    def diagonalize(self):
        eigenvalues, eigenvectors, is_diag=eigen_value_decomp(self.A)
        self.erank = get_effective_rank(self.A)
        self.cond = np.linalg.cond(eigenvectors)
        if is_diag:
            B_diag= np.linalg.inv(eigenvectors) @ self.B
            if self.meas=='scaled':
                eff_rank= 1+len(np.argwhere(np.abs(eigenvalues)>1.000001 )) 
            elif self.meas=='translated':
                 eff_rank= 1+len(np.argwhere(np.abs(eigenvalues)>0.0000001 ))     
            print("The", self.meas, self.fname, "SSM is diagonalizable with effective rank:", eff_rank, "\n")
            Translation= self.Fobj.F  @ self.Fobj.D.T
            eigenvalues= eigenvalues[0: eff_rank]
            eigenvectors=Translation @ eigenvectors[ :, 0:eff_rank ]
            B_diag= B_diag[0:eff_rank]       
            return eigenvalues.squeeze(), eigenvectors, B_diag.squeeze(), is_diag
        else:
            print("The", self.meas, self.fname,"SSM is non-diagonalizable with effective rank:", self.A.shape[0], "\n")
            return None, None, None, is_diag

    def plot_frame(self, which: str = "F", **kwargs):
        """
        Delegates basis plotting to the underlying Fobj instance.
        """
        if self.Fobj is None:
            raise AttributeError("SSM has no initialized Fobj.")
        self.Fobj.plot(which=which, **kwargs)

