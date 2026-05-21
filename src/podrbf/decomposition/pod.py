import numpy as np 

class POD:
    def __init__(self, data, n_modes):
        self.data = data
        self.n_modes = n_modes
        self.mean = np.mean(data, axis=0)
        self.centered_data = data - self.mean
        self.cov_matrix = np.cov(self.centered_data, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov_matrix)
        self.sort_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[self.sort_indices]
        self.eigenvectors = self.eigenvectors[:, self.sort_indices]
        self.modes = self.eigenvectors[:, :n_modes]
    
    def project(self, new_data):
        centered_new_data = new_data - self.mean
        return np.dot(centered_new_data, self.modes)    
    def reconstruct(self, coefficients):
        return np.dot(coefficients, self.modes.T) + self.mean  





