import numpy as np
import os.path as path
import torch
import os

class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def fit_tensor(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """

        self.mu = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        if type(data)==torch.Tensor:
            if type(self.mu) != torch.Tensor:
                self.mu = torch.tensor(self.mu, device=data.device)
                self.std = torch.tensor(self.std, device=data.device)
            else:
                pass
            result = (data - self.mu) / self.std
        elif isinstance(data, np.ndarray):
            if isinstance(self.mu, np.ndarray):
                result = (data - self.mu) / self.std
            else: # torch.tensor
                result = (data - self.mu.cpu().numpy()) / self.std.cpu().numpy()
        else:
            result=None
            raise NotImplementedError
        return result

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu

    def save_scaler(self, save_path):
        # torch version
        if isinstance(self.mu, np.ndarray) and isinstance(self.std, np.ndarray):
            mu_path = path.join(save_path, "mu.npy")
            std_path = path.join(save_path, "std.npy")
            np.save(mu_path, self.mu)
            np.save(std_path, self.std)
        # numpy version
        else:
            mu_path = path.join(save_path, "mu.pt")
            std_path = path.join(save_path, "std.pt")
            torch.save(self.mu, mu_path)
            torch.save(self.std, std_path)


    def load_scaler(self, load_path):
        load_funcs = {".npy": np.load, ".pt": torch.load}

        for ext, load_func in load_funcs.items():
            mu_path = os.path.join(load_path, "mu" + ext)
            std_path = os.path.join(load_path, "std" + ext)

            if os.path.exists(mu_path) and os.path.exists(std_path):
                self.mu = load_func(mu_path)
                self.std = load_func(std_path)
                return  # Return after successfully loading

        raise ValueError("No appropriate files (mu, std) found in the directory with .npy or .pt extension.")

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data
