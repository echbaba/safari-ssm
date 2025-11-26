import numpy as np

#This clas contains various synthetic signal generation dunctions.
class synthetic_signal:
    
    # this function just creates some toy data to test the functions.
    @staticmethod
    def generate_rand_data(L,smooth=False, smoothness=50):
        # generate a random string of data
        fval = np.random.rand(L,1)-0.5
        f = np.zeros([L,1])
        f[0] = fval[0]
        for i in range(1,L):
            f[i] = f[i-1] + fval[i]
        f = np.squeeze(f)
        if smooth==True:
            f = np.convolve(f, np.ones(smoothness)/smoothness, mode='same')
        return f
    
    @staticmethod
    def doppler(N):
        t=np.linspace(0.001, 0.999, N)
        return np.sqrt(t * (1 - t)) * np.sin(2.1 * np.pi / (t + 0.05))
    
    @staticmethod
    def bumps(N,heights,locations,widths):
        t=np.linspace(0.001, 0.999, N)
        X = np.zeros_like(t)
        for h, loc, w in zip(heights, locations, widths):
            X += h / ((1 + ((t - loc) / w)**2)**4)
        return X
    
    @staticmethod
    def blocks(N,heights,locations):
        t=np.linspace(0.001, 0.999, N)
        X = np.zeros_like(t)
        for i, (loc, height) in enumerate(zip(locations, heights)):
            X += height * (t >= loc)
        return X
    
    @staticmethod
    def piecewise_poly(N):
        t = np.linspace(0.001, 0.999, N)
        X = np.zeros_like(t)
        a=np.random.normal()
        b=np.random.uniform()
        c=np.random.uniform()
        X[t < 0.3] = a * t[t < 0.3]**2  # Quadratic on first segment
        X[(t >= 0.3) & (t < 0.6)] = b * (t[(t >= 0.3) & (t < 0.6)] - 0.3)**3 + 0.2  # Cubic on second segment
        X[t >= 0.6] = c  # Constant on third segment
        return X
    
    
    @staticmethod
    def random_piecewise_poly(N, locations):
        """
        Generate a piecewise polynomial over [0, 1] using NumPy.
        
        Args:
            N (int): Number of samples in the range [0, 1]
            locations (np.ndarray): 1D sorted array of K values in [0, 1] defining breakpoints
        
        Returns:
            np.ndarray: Array of shape (N,) with the piecewise polynomial values
        """
        t = np.linspace(0.0, 1.0, N)
        X = np.zeros_like(t)

        locations = np.sort(locations)
        # Add 0.0 and 1.0 as boundaries
        boundaries = np.concatenate([[0.0], locations, [1.0]])

        for i in range(len(boundaries) - 1):
            left = boundaries[i]
            right = boundaries[i + 1]
            mask = (t >= left) & (t < right if i < len(boundaries) - 2 else t <= right)

            degree = np.random.randint(0, 3)  # Randomly choose degree 1, 2, or 3
            local_t = t[mask] - left  # shift to [0, right-left]

            if degree == 0:
                a = np.random.randn()
                X[mask] = a
            elif degree == 1:
                a = np.random.randn()
                b = np.random.randn()
                X[mask] = a * local_t + b
            elif degree == 2:
                a = np.random.randn()
                b = np.random.randn()
                c = np.random.randn()
                X[mask] = a * local_t**2 + b * local_t + c

        return X
    
    
    @staticmethod
    def mishmash(N):
        t = np.linspace(0.001, 0.999, N)
        X = np.zeros_like(t)
        X[t < 0.3] = np.sin(2 * np.pi * 5 * t[t < 0.3])  # Sine wave on first segment
        X[(t >= 0.3) & (t < 0.7)] = t[(t >= 0.3) & (t < 0.7)]**2  # Quadratic on second segment
        X[t >= 0.7] = 3 * t[t >= 0.7] + 1  # Linear on third segment
        return X
    
    @staticmethod
    def spikes(N, spike_positions, spike_heights, spike_width=0.01):
        #t = np.linspace(0.001, 0.999, N)
        t = np.linspace(0.0, 1.0, N)
        X = np.zeros_like(t)
        for pos, height in zip(spike_positions, spike_heights):
            X[np.abs(t - pos) <= spike_width] = height  # Add spikes within a small interval around each position
        return X

    
