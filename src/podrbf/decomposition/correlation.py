import numpy as np 
import xarray as xr

class CrossCorrelation: 

    def __init__(self, dataset: xr.Dataset, field: str, chunksize: int = 1000):
        self.dataset = dataset
        self.field = field 
        self.chunksize = chunksize

        data_array = dataset[field]
        self.nsnaps = data_array.shape[-1]
        self.npoints = int(np.prod(data_array.shape[:-1]))

        self.C = None
        self.mean = None
        self.csd = None
    # ---------------------------------------------------------------------------------------- Public Functions
    def compute_covariance(self): 
        self.mu = self._mean() 
        self.C = self._covariance() 
        self.cond = self._condition_number() 
        return self.C, self.cond 
    
    def compute_csd(self):
        #cross spectral density
        return self.csd
    # ---------------------------------------------------------------------------------------- Private Functions
    def _mean(self):
        mu = np.zeros(self.N) 
        arr = self.dataset[self.field].values.reshape(self.n_points, self.N)
        for start, end in self._chunks():
            mu += arr[start:end, :].sum(axis=0)

        return mu / self.N
    
    def _covariance(self):
        """
        C = X'.T @ X'

        where X'[k, :] = X[k, :] - mu

        Accumulated chunk-wise:
            C += X'[start:end, :].T @ X'[start:end, :]
        """
        C   = np.zeros((self.N, self.N))
        arr = self.dataset[self.field].values.reshape(self.n_points, self.N)

        for start, end in self._chunks():
            Xp  = arr[start:end, :] - self.mu 
            C  += Xp.T @ Xp

        return C / (self.N - 1)

    def _chunks(self):
        for start in range(0, self.n_points, self.chunk_size):
            yield start, min(start + self.chunk_size, self.n_points)

    def _condition_number(self):
        