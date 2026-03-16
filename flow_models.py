import normflows as nf
from math import ceil # only used in RealNVP_Flow
import torch
import numpy as np
from torchkde import KernelDensity
from sklearn.model_selection import KFold


class KDE_Estimator:
    """KDE Estimator with log_prob and sample methods"""

    def __init__(self, latent_size, device='cuda:0'):
        self.latent_size = int(latent_size)
        self.device = torch.device(device)
        self.kde = None
        self.best_bw = None
        self._train_data = None
        self.is_fitted = False

    def fit(self, x, cv_folds=20):
        if (self.is_fitted):
            print("Warning: KDE_Estimator is already fitted. Refitting will overwrite the existing model.")

        bw_values = np.logspace(-2, 1, 20)

        x_np = x.detach().cpu().numpy()
        best_score = -np.inf
        best_bw = None

        for bw in bw_values:
            scores = []
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=3043)
            for train_idx, test_idx in kf.split(x_np):
                train_data = torch.from_numpy(x_np[train_idx]).to(dtype=torch.float32, device=self.device)
                test_data = torch.from_numpy(x_np[test_idx]).to(dtype=torch.float32, device=self.device)
                kde = KernelDensity(bandwidth=bw, kernel='gaussian').fit(train_data)
                scores.append(kde.score_samples(test_data).mean().item())

            avg_score = scores.mean()
            if avg_score > best_score:
                best_score = avg_score
                best_bw = bw

        self.best_bw = best_bw
        self._train_data = x
        self.kde = KernelDensity(bandwidth=self.best_bw, kernel='gaussian').fit(self._train_data)
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
        log_prob = self.kde.score_samples(samples)
        return samples, log_prob

    # def to(self, device):
    #     self.device = torch.device(device)
    #     if self._train_data is not None:
    #         self._train_data = self._train_data.to(self.device)
    #     if self.is_fitted and self.best_bw is not None and self._train_data is not None:
    #         self.kde = KernelDensity(bandwidth=self.best_bw, kernel='gaussian').fit(self._train_data)
    #     return self

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
    for _ in range(16):
        param_map = nf.nets.MLP([ceil(int(latent_size) / 2), hidden_units, hidden_units, latent_size], init_zeros=True, output_fn='sigmoid')
        # Add flow layer
        flows += [nf.flows.AffineCoupling(param_map)]
        
        # Swap dimensions
        if (latent_size > 1):
            flows += [nf.flows.Permute(latent_size, mode='swap')]

    # Latent space
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)

    if (mean is not None):
        q0.loc.data = mean.reshape(1, -1).to(dtype=torch.float32)
    if (std is not None):
        q0.log_scale.data = std.reshape(1, -1).to(dtype=torch.float32).log()
    
    # construct flow model
    return nf.NormalizingFlow(q0=q0, flows=flows).to(device)

# def learn_1d_pdf(data, cv_folds=20):
#     """
#     Learn a 1D probability density function using scipy's KDE estimator with cross-validated bandwidth.

#     Args:
#         data (np.ndarray): Shape (N,) array of 1D data points to fit the PDF to.
#         cv_folds (int): Number of cross-validation folds.

#     Returns:
#         kde (gaussian_kde): Fitted KDE object with best bandwidth.
#     """
#     data = np.asarray(data).reshape(-1)
#     if bw_values is None:
#         bw_values = np.logspace(-2, 1, 20)

#     best_score = -np.inf
#     best_bw = None

#     for bw in bw_values:
#         scores = []
#         kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
#         for train_idx, test_idx in kf.split(data):
#             train_data = data[train_idx]
#             test_data = data[test_idx]
#             kde = gaussian_kde(train_data, bw_method=bw)
#             # log-likelihood on test data
#             log_likelihood = np.mean(np.log(kde.evaluate(test_data) + 1e-12))
#             scores.append(log_likelihood)
#         avg_score = np.mean(scores)
#         if avg_score > best_score:
#             best_score = avg_score
#             best_bw = bw

#     kde = gaussian_kde(data, bw_method=best_bw)
#     return kde
