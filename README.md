first commit

in 
/home/jchang/miniconda3/envs/flow/lib/python3.11/site-packages/torchkde/modules.py
in `sample` function, move torch.randn(...) to the same device as data
X = self.bandwidth * torch.randn(n_samples, data.shape[1]).to(data.device) + data[idxs]

in 
/home/jchang/miniconda3/envs/flow/lib/python3.11/site-packages/torchkde/modules.py
in function `score_samples`, add a minimum of 1e-8 when estimating density 
log_density.append((density + 1e-8 ** X.shape[1]).log())