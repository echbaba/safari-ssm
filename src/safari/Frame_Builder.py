import numpy as np
import math
import findiff as fd
import pywt
import tabulate
import warnings
import matplotlib.pyplot as plt

def framelist():
# option, keyword, default value, description
    opts = [
        ["Frame Name", "Keyword Value", "Addl. Opts", "Default Vals", "Notes"],
        ["Legendre Polynomials", "legendre", "N/A", "N/A", "Not stably diagonalizable for N > 20 \nUses HiPPO closed-form solution"],
        ["Fourier Basis", "fourier", "N/A", "N/A", "Rank is N+1 due to DC term"],
        ["Chebyshev Polynomials", "chebyshev", "N/A", "N/A", "N/A"],
        ["Laguerre Polynomials", "laguerre", "N/A", "N/A", "N > 100 may be unstable"],
        ["Bernstein Polynomials", "bernstein", "N/A", "N/A", "N/A"],
        ["Gabor Filters [LEGACY]", "[N/A]", "fmin, fmax \nredundancy", "fmin=-2.0, fmax=1.0 \nredundancy=1.0", "Fmin, Fmax = 2**fmin (0.25), 2**fmax (2.0). \n[Deprecated Sept 2026.]"],
        ["Gabor Filters", "gabor", "freqs, redundancy, \nsigma_factor, base_freq, \nnum_freq_levels, freq_scale, \nmultiscale", 
         "freqs=None, redundancy=0.0, \nsigma_factor=0.4, base_freq=8, \nnum_freq_levels=4, freq_scale='dyadic', \nmultiscale=False", 
         "Automatic allocation: set base_freq, num_freq_levels.\nExplicit frequencies: pass list eg [4,8,16] to freqs. \nSigma factor is for Gaussian envelope.\n"
         "freq_scale can be 'dyadic' or 'linear'. \nMultiscale=True narrows envelope at higher frequencies."],
        ["Daubechies Wavelets", "daubechies", "dborder", "db11", "DWT construction via PyWavelets. \nRank is N due to orthogonality."]       
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
        self.N = params.get("N", 32)    #if N is provided, use it, otherwise use the default value of 10
        self.L = params.get("L", 2**10)   #similar to the above
        self.range_min = params.get("range_min", 0.0)
        self.range_max = params.get("range_max", 1.0)
        # SPECIFIC TO GABOR FRAME
        self.freqs = params.get("freqs", None)  # only relevant for gabor frame
        self.sigma_factor = params.get("sigma_factor", 0.4)  # only relevant for gabor frame
        self.base_freq = params.get("base_freq", 8)  # only relevant for gabor frame
        self.num_freq_levels = params.get("num_freq_levels", 4)  # only relevant for gabor frame
        self.freq_scale = params.get("freq_scale", 'dyadic')  # only relevant for gabor frame
        self.multiscale = params.get("multiscale", False)  # only relevant for gabor frame
        self.redundancy = params.get("redundancy", 0.0) # only relevant for gabor frame
        # SPECIFIC TO LEGACY GABOR FRAME
        self.m = params.get("m", 4)  # only relevant for legacy gabor frame
        # SPECIFIC TO DAUBECHIES FRAME
        self.dborder = params.get("dborder", 'db6') # only relevant for daubechies frame
        self.rcond = params.get("rcond", 0.01) 
        self.F = params.get("F", None)
        self.dF = params.get("dF", None)
        self.D = params.get("D", None)
        if self.F is None:
            self.generateFrame()
        else:
            self.make_frame(F = self.F, dF = self.dF, D = self.D)
        self.completeFrame()  # compute dual and derivative if not already provided

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
      
    def generateFrame(self):
    # input:    N, number of coefficients
    #           L, length of basis of signal (only matters for numerical accuracy)
    #           type
    # output:   F, an NxL frame (or basis) on interval
    #           D, the dual frame (only if analytical solution available)
    #           dF, the derivative of the frame (only if analytical solution available)
        fname = self.fname
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

            if self.N % 2 != 0:
                raise ValueError("N must be even for real/imag (cosine/sine) pairs.")

            num_complex_atoms = self.N // 2

            # 1. Determine frequency levels
            if self.freqs is not None:
                frequency_levels = list(self.freqs)
            elif self.freq_scale == 'dyadic':
                frequency_levels = [self.base_freq * (2 ** j) for j in range(self.num_freq_levels)]
            elif self.freq_scale == 'linear':
                frequency_levels = [self.base_freq * (j + 1) for j in range(self.num_freq_levels)]
            else:
                raise ValueError("freq_scale must be 'dyadic' or 'linear'.")

            num_levels = len(frequency_levels)
            # Number of spatial translations per frequency level
            num_positions = max(1, int(np.ceil(num_complex_atoms / num_levels)))

            t = np.linspace(0, 1, self.L, endpoint=False)
            dx = 1.0 / num_positions
            centers = np.linspace(dx / 2.0, 1.0 - dx / 2.0, num_positions)
            
            # Base spatial window width
            base_sigma = dx * self.sigma_factor * (1.0 + self.redundancy)

            F = np.zeros((self.N, self.L), dtype=np.float64)
            row_idx = 0

            # 2. Iterate across frequency levels and spatial translations
            for f_idx, f in enumerate(frequency_levels):
                # In multiscale/wavelet mode, envelope narrows at higher frequencies
                if self.multiscale:
                    scale_ratio = frequency_levels[0] / f
                    sigma = base_sigma * np.sqrt(scale_ratio)
                else:
                    sigma = base_sigma

                for center in centers:
                    if row_idx >= self.N:
                        break

                    # Distance to center with periodic boundary wrapping
                    dt = np.abs(t - center)
                    dt = np.minimum(dt, 1.0 - dt)
                    
                    # Localized Gaussian envelope
                    envelope = np.exp(-0.5 * (dt / sigma) ** 2)
                    envelope[envelope < 1e-4] = 0.0  # Compact support truncation

                    # Centered carrier wave
                    phase = 2.0 * np.pi * f * (t - center)
                    cos_atom = envelope * np.cos(phase)
                    sin_atom = envelope * np.sin(phase)

                    # L2 Energy Normalization
                    norm_c = np.linalg.norm(cos_atom)
                    norm_s = np.linalg.norm(sin_atom)

                    F[row_idx, :] = cos_atom / (norm_c if norm_c > 0 else 1.0)
                    row_idx += 1

                    if row_idx < self.N:
                        F[row_idx, :] = sin_atom / (norm_s if norm_s > 0 else 1.0)
                        row_idx += 1

                if row_idx >= self.N:
                    break

            D = np.linalg.pinv(F, rcond=self.rcond).T

        elif fname=='gabor_legacy':
            a = (1-self.redundancy) / (2**0.5) # time shift
            b = 1 / (2**0.5) # frequency shift
            Nnew = self.N//2
            t = np.linspace(0,Nnew*a,self.L)
            for i in range(Nnew):
                g = np.exp(2.0*1j*np.pi*self.m*b*t)*(2**(0.25))*np.exp(-np.pi*(t-i*a)**2)
                F[2*i,:] = np.real(g)  # normalize each filter to unit L1 norm
                F[2*i+1,:] = np.imag(g)  # normalize each filter to unit L1 norm
            D = np.linalg.pinv(F, rcond=self.rcond).T

        elif fname=='daubechies':
            if not is_power_of_two(self.N):
                new_N = nearest_power_of_two(self.N)
                warnings.warn(
                    f"Input N={self.N} is not a power of 2. Adjusted to nearest power of 2: {new_N}.",
                    UserWarning
                )
                self.N = new_N
            if not is_power_of_two(self.L):
                new_L = nearest_power_of_two(self.L)
                warnings.warn(
                    f"Input L={self.L} is not a power of 2. Adjusted to nearest power of 2: {new_L}.",
                    UserWarning
                )
                self.L = new_L

            wavelet = pywt.Wavelet(self.dborder)
            print(f"wavelet info: {wavelet}")

            F = np.zeros((self.N, self.L), dtype=np.float64)
            # Determine maximum decomposition level for length N

            # Max decomposition level for resolution L
            # We decompose all the way down to a 1-sample approximation
            total_levels = int(np.log2(self.L))
            
            # Create the coefficient structure for decomposition down to length 1
            # Structure: [cA_0 (len 1), cD_0 (len 1), cD_1 (len 2), cD_2 (len 4), ..., cD_{total_levels-1} (len L/2)]
            coeffs_template = [np.zeros(1, dtype=np.float64)]  # cA (Father)
            for j in range(total_levels):
                coeffs_template.append(np.zeros(2 ** j, dtype=np.float64))  # cD_j

            row_idx = 0

            # 1. Father wavelet (Row 0: Unit impulse in the coarsest approximation subband)
            impulse_coeffs = [c.copy() for c in coeffs_template]
            impulse_coeffs[0][0] = 1.0
            F[row_idx, :] = pywt.waverec(impulse_coeffs, wavelet=wavelet, mode='periodization')
            row_idx += 1

            # 2. Mother & Detail Wavelets (Rows 1 to N-1)
            # Traverse subbands from coarsest detail (len 1) up to the required N-1 total details
            for band_idx in range(1, len(coeffs_template)):
                band_len = len(coeffs_template[band_idx])
                for k in range(band_len):
                    if row_idx >= self.N:
                        break
                    impulse_coeffs = [c.copy() for c in coeffs_template]
                    impulse_coeffs[band_idx][k] = 1.0
                    
                    # Periodic inverse DWT guarantees exact isometry / orthonormality
                    F[row_idx, :] = pywt.waverec(impulse_coeffs, wavelet=wavelet, mode='periodization')
                    row_idx += 1
                    
                if row_idx >= self.N:
                    break


            D = np.linalg.pinv(F, rcond=self.rcond).T  # Dual frame via pseudoinverse
            # wavelet = pywt.Wavelet(self.dborder)
            # print(f"Generating Daubechies frame with wavelet '{self.dborder}' for N={self.N}, L={self.L}.")
            # max_level = pywt.dwt_max_level(data_len=self.N, filter_len=wavelet.dec_len)
            # if max_level == 0:
            #     raise ValueError(
            #         f"N={self.N} is too small for wavelet '{self.dborder}' with filter length {wavelet.dec_len}."
            #     )
            
            # # Generate canonical basis impulses in coefficient space and reconstruct
            # dummy_signal = np.zeros(self.N)
            # coeffs_structure = pywt.wavedec(dummy_signal, wavelet=wavelet, level=max_level, mode='periodization')
            # row_idx = 0
            # # Generate Daubechies wavelet frame using pywt
            # for band_idx, band in enumerate(coeffs_structure):
            #     band_len = len(band)
            #     for coeff_pos in range(band_len):
            #         if row_idx >= self.N:
            #             break
            #         # Create a zeroed copy of coefficient structure
            #         impulse_coeffs = [np.zeros_like(c) for c in coeffs_structure]
            #         impulse_coeffs[band_idx][coeff_pos] = 1.0
                    
            #         # Synthesize standard length-N basis function via IDWT
            #         basis_elem = pywt.waverec(impulse_coeffs, wavelet=wavelet, mode='periodization')
                    
            #         # Resample or pad/truncate to target length L
            #         if len(basis_elem) != self.L:
            #             # Interpolate to match required discretized length L
            #             x_old = np.linspace(0, 1, len(basis_elem), endpoint=False)
            #             x_new = np.linspace(0, 1, self.L, endpoint=False)
            #             basis_elem_resampled = np.interp(x_new, x_old, basis_elem)
            #             # Normalize energy after interpolation
            #             norm = np.linalg.norm(basis_elem_resampled)
            #             if norm > 0:
            #                 basis_elem_resampled /= norm
            #             F[row_idx, :] = basis_elem_resampled
            #         else:
            #             F[row_idx, :] = basis_elem[:self.L]
            #         row_idx += 1
            

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

    def completeFrame(self):
        # if a dual of the frame has been provided, use it.  Else, compute it.
        if self.D is None:
            #print("Computing dual frame")
            self.D = np.linalg.pinv(self.F).T
        # if the derivative of the frame has been provided, use it.  Else, compute it.
        if self.dF is None:
            #print("Computing derivative of frame")
            self.dF = self.fdFrame(self.F)

    def fdFrame(self, F):
    # input:    F, the frame
    # output:   dF, first derivative of the frame calculated by finite difference method
        dx = 1/self.L
        dF_dx = fd.FinDiff(1,dx)
        dF = dF_dx(F)
        return dF

    def plot(
        self, 
        which: str = "F", 
        spacing: float = 1.3, 
        max_functions: int = 32, 
        title: str = None
    ):
        """
        Plots basis functions offset vertically on the same axes.
        
        Parameters
        ----------
        which : str
            Which frame matrix to plot: 'F' (frame), 'D' (dual), or 'dF' (derivative).
        spacing : float
            Vertical spacing multiplier between baseline offsets.
        max_functions : int
            Maximum number of basis functions to render.
        title : str, optional
            Custom plot title.
        """
        mat = getattr(self, which, None)
        if mat is None:
            raise ValueError(f"Matrix '{which}' is not computed or available on Fobj.")

        N_total, L = mat.shape
        N = min(N_total, max_functions)
        t = np.linspace(self.range_min, self.range_max, L)

        fig, ax = plt.subplots(figsize=(10, max(5, N * 0.45)))
        max_amp = np.max(np.abs(mat[:N, :]))
        offset_step = (max_amp if max_amp > 0 else 1.0) * spacing
        colors = plt.cm.viridis(np.linspace(0, 0.9, N))

        for i in range(N):
            baseline = (N - 1 - i) * offset_step
            y = mat[i, :] + baseline
            ax.axhline(baseline, color='lightgray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.plot(t, y, color=colors[i], linewidth=1.2)

        ax.set_yticks([(N - 1 - i) * offset_step for i in range(N)])
        ax.set_yticklabels([f"$\psi_{{{i}}}$" for i in range(N)])
        ax.set_xlabel("Time ($t$)")
        ax.set_ylabel("Basis Index")
        
        default_title = f"{self.fname.capitalize()} Basis '{which}' (N={N}/{N_total}, L={L})"
        ax.set_title(title if title is not None else default_title)
        ax.grid(True, axis='x', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

# HELPER FUNCTIONS FOR DAUBECHIES FRAME GENERATION
def nearest_power_of_two(n: int) -> int:
    """Helper function to find the nearest power of 2 for a given integer."""
    if n <= 0:
        return 1
    lower = 2 ** int(np.floor(np.log2(n)))
    upper = 2 ** int(np.ceil(np.log2(n)))
    return lower if (n - lower) <= (upper - n) else upper

def is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0





