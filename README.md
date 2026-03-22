first commit

in 
/home/jchang/miniconda3/envs/flow/lib/python3.11/site-packages/torchkde/modules.py
change one line in `sample` function: 
X = self.bandwidth * torch.randn(n_samples, data.shape[1]).to(data.device) + data[idxs]

in 
/home/jchang/miniconda3/envs/flow/lib/python3.11/site-packages/torchkde/modules.py
in function `score_samples`
add 1e-8, change to 
log_density.append((density + 1e-8 ** X.shape[1]).log())