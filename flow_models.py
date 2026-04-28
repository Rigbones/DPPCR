import normflows as nf
from math import ceil # only used in RealNVP_Flow
import torch
import numpy as np
from torchkde import KernelDensity
from sklearn.model_selection import KFold
from scipy.stats import gaussian_kde


class KDE_Estimator:
    """KDE Estimator"""

    def __init__(self, latent_size, device='cuda:0'):
        """
        Args:
            latent_size (int): Dimensionality of the latent space.
        """
        self.latent_size = latent_size
        self.device = device
        self.kde = None
        self.bw = None
        self.is_fitted = False

    def _get_scipy_scale(self, x):
        """Convert scipy-like factor to an isotropic absolute bandwidth scale.
        Must return float
        Args:
            x (torch.Tensor): Shape (N, latent_size) tensor
        Returns:
            float: Absolute bandwidth scale for isotropic Gaussian kernel.
        """

        # if latent space is 1D, use the std of the data as the scale
        if self.latent_size == 1:
            scale = x.flatten().std(unbiased=True) # would have error if x only has 1 element
        # else if latent space is 2D or higher
        else:
            num_samples = x.shape[0]
            centered = x - x.mean(axis=0, keepdim=True)
            cov = (centered.T @ centered) / (num_samples - 1)
            scale = (torch.trace(cov) / self.latent_size).sqrt()

        return float(scale.clamp_min(1e-8).item())

    def fit(self, x, cv_folds=10):
        """
        Args:
            x (torch.Tensor): Shape (N, latent_size) tensor of data points to fit.
            cv_folds (int): Number of cross-validation folds for bandwidth selection.
        """
        if (self.is_fitted):
            print("Warning: KDE_Estimator is already fitted. Refitting will overwrite the existing model.")

        x = x.to(dtype=torch.float32, device=self.device)

        best_score = -float('inf')
        best_factor = None

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scale = self._get_scipy_scale(x)
        # Adapt factor range smoothly: smaller std then shift search higher
        # This prevents underflow without hardcoded thresholds
        shift = 1.0 * (1.0 - np.log10(scale + 1.0))
        log_factor_min = -2.0 + shift
        log_factor_max = 1.0 + shift
        factors = np.logspace(log_factor_min, log_factor_max, 20).tolist()

        for factor in factors:

            # calculate average score over all folds
            scores = []
            for idx_train, idx_test in kf.split(x):
                x_train = x[idx_train]
                x_test = x[idx_test]
                bw = factor * self._get_scipy_scale(x_train)

                kde = KernelDensity(bandwidth=bw, kernel='gaussian').fit(x_train)
                scores.append(kde.score_samples(x_test).mean().item())
            score = sum(scores) / len(scores)

            # update best bandwidth if needed
            if score > best_score:
                best_score = score
                best_factor = factor

        self.bw = float(best_factor * self._get_scipy_scale(x))
        self.kde = KernelDensity(bandwidth=self.bw, kernel='gaussian').fit(x)
        self.is_fitted = True
        return self

    def log_prob(self, x):
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before log_prob(...).")
        return self.kde.score_samples(x)

    def sample(self, num_samples):
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before sample(...).")
        samples = self.kde.sample(num_samples)
        log_prob = self.kde.score_samples(samples.to(device=self.device))
        return samples, log_prob # we return both samples and log_prob for consistency with normflows package

def Lipschitz_Flow(latent_size, mean=None, std=None, device='cuda:0'):
    hidden_units = 64
    flows = []
    for _ in range(16):
        net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] + [hidden_units] + [latent_size], init_zeros=True, lipschitz_const=0.9)
        flows += [nf.flows.Residual(net, reduce_memory=True, exact_trace=True)] # exact_trace=True is important to avoid a fluctuating logdetJ
        flows += [nf.flows.ActNorm(latent_size)]

    # Latent space
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)

    if (mean is not None):
        q0.loc.data = mean.reshape(1, -1).to(dtype=torch.float32)
    if (std is not None):
        q0.log_scale.data = std.reshape(1, -1).to(dtype=torch.float32).log()
    
    # construct flow model
    return nf.NormalizingFlow(q0=q0, flows=flows).to(device)

def RealNVP_Flow(latent_size, mean=None, std=None, device='cuda:0'):
    hidden_units = 64
    flows = []
    for _ in range(32):
        param_map = nf.nets.MLP([ceil(int(latent_size) / 2), hidden_units, hidden_units, latent_size], init_zeros=True, output_fn='sigmoid')
        # Add flow layer
        flows += [nf.flows.AffineCouplingBlock(param_map)]
        # Swap dimensions
        flows += [nf.flows.Permute(latent_size, mode='swap')]
        flows += [nf.flows.ActNorm(latent_size)]

    # Latent space
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)

    if (mean is not None):
        q0.loc.data = mean.reshape(1, -1).to(dtype=torch.float32)
    if (std is not None):
        q0.log_scale.data = std.reshape(1, -1).to(dtype=torch.float32).log()
    
    # construct flow model
    return nf.NormalizingFlow(q0=q0, flows=flows).to(device)

def learn_1d_pdf(data, cv_folds=20, bw_values=None):
    """
    Learn a 1D probability density function using scipy's KDE estimator with cross-validated bandwidth.

    Args:
        data (np.ndarray): Shape (N,) array of 1D data points to fit the PDF to.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        kde (gaussian_kde): Fitted KDE object with best bandwidth.
    """
    data = np.asarray(data).reshape(-1)
    if bw_values is None:
        bw_values = np.logspace(-2, 1, 20)

    best_score = -np.inf
    best_bw = None

    for bw in bw_values:
        scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]
            kde = gaussian_kde(train_data, bw_method=bw)
            # log-likelihood on test data
            log_likelihood = np.mean(np.log(kde.evaluate(test_data) + 1e-12))
            scores.append(log_likelihood)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_bw = bw

    kde = gaussian_kde(data, bw_method=best_bw)
    return kde
