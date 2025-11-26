import numpy as np
import math
import findiff as fd
import pywt
import tabulate

def framelist():
# option, keyword, default value, description
    opts = [
        ["Frame Name", "Keyword Value", "Addl. Opts", "Default Vals", "Notes"],
        ["Legendre Polynomials", "legendre", "N/A", "N/A", "Not stably diagonalizable for N > 20"],
        ["Fourier Basis", "fourier", "N/A", "N/A", "Rank is N+1 due to DC term"],
        ["Chebyshev Polynomials", "chebyshev", "N/A", "N/A", "N/A"],
        ["Laguerre Polynomials", "laguerre", "N/A", "N/A", "N > 100 may be unstable"],
        ["Bernstein Polynomials", "bernstein", "N/A", "N/A", "N/A"],
        ["Gabor Filters", "gabor", "fmin, fmax \nredundancy", "fmin=-2.0, fmax=1.0 \nredundancy=1.0", "Fmin, Fmax = 2**fmin (0.25), 2**fmax (2.0). \nFor redundancy>1, rank may be less than N."],       
    ]
    headers = opts[0]
    table = opts[1:]
    print(tabulate.tabulate(table, headers, tablefmt="grid"))

class Fobj:
    """
    Frame object contains the N x L frame (or basis) F, 
    the dual frame D, and the derivative of the frame dF.
    """

    def __init__(self, **params):
        self.fname = params.get("fname", 'custom')
        self.N = params.get("N", 10)    #if N is provided, use it, otherwise use the default value of 10
        self.L = params.get("L", 2**10)   #similar to the above
        self.range_min = params.get("range_min", 0.0)
        self.range_max = params.get("range_max", 1.0)
        # self.fmin = params.get("fmin", -2.0) # only relevant for gabor frame
        # self.fmax = params.get("fmax", 1.0)  # only relevant for gabor frame
        self.redundancy = params.get("redundancy", 0.0) # only relevant for gabor frame
        self.m = params.get("m", 1)  # only relevant for gabor frame
        self.rcond = params.get("rcond", 0.01) 
        self.F = params.get("F", None)
        self.dF = params.get("dF", None)
        self.D = params.get("D", None)
        if self.F is None:
            self.generateFrame(fname=self.fname)
        else:
            self.make_frame(F = self.F, dF = self.dF, D = self.D)

    def make_frame(self, **args):
        self.F = args.get("F")
        self.dF = args.get("dF")
        self.D = args.get("D")
        self.N, self.L = self.F.shape
        dF = np.empty((self.N, self.L), dtype=float)
        d_dt=fd.FinDiff(0,1/self.L)   #prepares  the finite difference module to fidn the dervative of Frame
        if self.dF is not None:
            for i in range(self.N):
                dF[i,:]= d_dt(self.F[i, :])
        if self.D is not None:
            self.D= np.linalg.pinv(self.F, rcond=self.rcond).T 
      
    def generateFrame(self, fname):
    # input:    N, number of coefficients
    #           L, length of basis of signal (only matters for numerical accuracy)
    #           type
    # output:   F, an NxL frame (or basis) on interval
    #           D, the dual frame (only if analytical solution available)
    #           dF, the derivative of the frame (only if analytical solution available)

        if fname=='legendre':        
            F = np.zeros([self.N,self.L])
            norm = np.sqrt(2*np.arange(self.N)+1)[:,None] # normalization vector for scaled legendre
            # np.polynomial.legendre generates a *single polynomial* with coefficients for degree
            # Eg, passing (1,2,3) gives you P = 1P0 + 2P1 + 3P2.
            # We want to keep each one separate for the basis, so we'll generate them individually.
            for i in range(self.N):
                # To get a Legendre polynomial of order 2, we can pass (0,0,1,0), to get order 3 we'd pass (0,0,0,1)...
                # I will combine this with the scaling as (0,0,c2,0), (0,0,0,c3), etc.
                coef = np.zeros(self.N,)
                coef[i] = norm[i]
                # Legendre polynomial object with coefficients (0,0,...c_i,..0)
                # evaluated on [-1,1] and mapped to [0,1]
                # p = np.polynomial.legendre.Legendre(coef,[0,1],[-1,1]) 
                p = np.polynomial.legendre.Legendre.basis(i, [self.range_min, self.range_max], [-1, 1])
                # p is a polynomial object and we want a vector
                (x,y) = p.linspace(self.L) # evaluate over domain [0,1] at L points 
                # scaling by sqrt(2n+1) to make orthonormal over [0,1]
                F[i,:] = y * ((2*i+1)**0.5)
            D = F / self.L # orthogonal basis, so the dual (inverse) is itself.
            # we've implicitly scaled F by L though, so we need to divide by L in the inverse.
        
        elif fname=='fourier':
            lvl = self.N//2
            F= np.zeros((1+2*lvl,self.L))
            D= np.zeros((1+2*lvl,self.L))
            dF=np.zeros((1+2*lvl,self.L))
            x= np.arange(self.L)/self.L
            F[0:self.L]= 1
            D[0:self.L]= 1
            for i in range(lvl):
                F[2*i+1,:]= 2**0.5 * np.cos( 2*np.pi * (i+1) * x )
                F[2*i+2,:]= 2**0.5 * np.sin( 2*np.pi * (i+1) * x )
                # derivative is available analytically, so we will produce it here.
                dF[2*i+1,:]= -2**0.5 * 2*np.pi*(i+1) * np.sin( 2*np.pi * (i+1) * x ) 
                dF[2*i+2,:]= 2**0.5 * 2*np.pi*(i+1) * np.cos( 2*np.pi * (i+1) * x ) 
            D = F / self.L # orthogonal basis, so the dual (inverse) is itself.

        elif fname=='chebyshev':
            x = np.linspace(-1, 1, self.L)
            F = np.empty((self.N, self.L), dtype=float)
            for i in range(self.N):
                Ti = np.polynomial.chebyshev.Chebyshev.basis(i)     # T_i
                F[i, :] = Ti(x)
            D = np.linalg.pinv(F, rcond=self.rcond).T 

        elif fname=='laguerre':
            dmax = self.N*5 # heuristically -- need 10x domain to see convergence
            x = np.linspace(0, dmax, self.L)
            F = np.empty((self.N, self.L), dtype=float)
            for i in range(self.N):
                Li = np.polynomial.laguerre.Laguerre.basis(i,domain=[0,dmax],window=[0,dmax])     # L_i
                tmp = Li(x)
                F[i, :] = tmp * np.exp(-x/2)  # orthogonality imposed
            D = np.linalg.pinv(F, rcond=self.rcond).T

        elif fname=='bernstein':
            x = np.linspace(0, 1, self.L)
            F = np.empty((self.N, self.L), dtype=float)
            n = self.N - 1  # Bernstein polynomials are degree n with n+1 basis functions
            for i in range(self.N):
                coeff = math.comb(n, i)
                F[i, :] = coeff * (x**i) * ((1 - x)**(n - i))
            D = np.linalg.pinv(F, rcond=self.rcond).T

        elif fname=='gabor':
            F = np.empty((self.N, self.L), dtype=float)
            D = np.empty((self.N, self.L), dtype=float)
            a = (1-self.redundancy) / (2**0.5) # time shift
            b = 1 / (2**0.5) # frequency shift
            Nnew = self.N//2
            t = np.linspace(0,Nnew*a,self.L)
            for i in range(Nnew):
                g = np.exp(2.0*1j*np.pi*self.m*b*t)*(2**(0.25))*np.exp(-np.pi*(t-i*a)**2)
                F[2*i,:] = np.real(g)  # normalize each filter to unit L1 norm
                F[2*i+1,:] = np.imag(g)  # normalize each filter to unit L1 norm
            D = np.linalg.pinv(F, rcond=self.rcond).T

        else:
            raise ValueError("Unknown frame type for built-in frame generator.")
    
        self.F = F
        try:
            dF
            self.dF = dF
        except:
            self.dF = None
        try:
            D
            self.D = D
        except:
            self.D = None

