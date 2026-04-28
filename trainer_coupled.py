# arrays, ML related
import normflows as nf
import numpy as np
import torch
import torch.nn as nn

# python libraries
from tqdm import tqdm
import matplotlib.pyplot as plt

from flow_models import RealNVP_Flow, Lipschitz_Flow, KDE_Estimator
from visualize import visualize

from pytorch3d.loss import chamfer_distance

def learn_pdf(x, epochs=3000, batch_size=150_000, model='RealNVP', save=None, device='cuda:0'):
    """
    Args:
        model (str): either 'RealNVP', 'Lipschitz' or 'KDE'
        epochs: number of epochs to train for, only applies to RealNVP and Lipschitz
        batch_size: batch size to use during training, only applies to RealNVP and Lipschitz
    """
    ### set up model ###
    if model == 'RealNVP':
        nfm = RealNVP_Flow(latent_size=x.shape[-1], mean=x.mean(axis=0), std=x.std(axis=0), device=device)
    elif model == 'Lipschitz':
        nfm = Lipschitz_Flow(latent_size=x.shape[-1], mean=x.mean(axis=0), std=x.std(axis=0), device=device)
    elif model == 'KDE':
        nfm = KDE_Estimator(latent_size=x.shape[-1], device=device).fit(x)
        return nfm
    optimizer = torch.optim.AdamW(nfm.parameters(), lr=5e-4)

    ### training ###
    progress_bar = tqdm(range(epochs))

    for epoch in progress_bar:
        optimizer.zero_grad()

        # get a random batch from x
        indices = torch.randperm(x.shape[0])[:batch_size] # shape (batch_size, latent_size)
        x_batch = x[indices]

        # Compute loss. forward_kld estimates E_{x ~ x_batch}[-log p_theta(x)]
        loss = nfm.forward_kld(x_batch)
        
        # Do backprop and optimizerizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        # Make layers Lipschitz continuous
        if (model == 'Lipschitz'):
            nf.utils.update_lipschitz(nfm, 50)
        
        # Print loss
        # progress_bar.set_postfix({'Loss': f"{loss.item():.2f}", 'lr': f"{scheduler.get_last_lr()[0]:.7f}"})
        progress_bar.set_postfix({'Loss': f"{loss.item():.2f}"})

        # save intermediate results every 50 epochs
        if save is not None and ((epoch % 50 == 0) or epoch < 30):
            with torch.no_grad():
                samples = nfm.sample(15_000)[0] # returns (samples, log_prob), we need samples
                if (save.endswith('.npy')):
                    np.save(save, samples.detach().cpu().numpy())
                    print(f"Epoch {epoch}: Saved intermediate results to {save}")
                elif (save.endswith('.txt')):
                    np.savetxt(save, samples.detach().cpu().numpy())
                    print(f"Epoch {epoch}: Saved intermediate results to {save}")
                else:
                    print(f"Unsupported save format {save}, expected .npy or .txt. Skipping saving intermediate results.")

    if save is not None:
        torch.save(nfm.state_dict(), f"{save}.pth")

    return nfm

class Differential(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.x = nn.Parameter(torch.tensor([0.0], device=device))
        self.y = nn.Parameter(torch.tensor([0.0], device=device))
        self.z = nn.Parameter(torch.tensor([0.0], device=device))
        self.r = nn.Parameter(torch.tensor([0.0], device=device))
        self.tx = nn.Parameter(torch.tensor([0.0], device=device))
        self.ty = nn.Parameter(torch.tensor([0.0], device=device))
        self.tz = nn.Parameter(torch.tensor([0.0], device=device))

    def forward(self):
        raise NotImplementedError
    
    def zero_(self):
        self.x.data.zero_()
        self.y.data.zero_()
        self.z.data.zero_()
        self.r.data.zero_()
        self.tx.data.zero_()
        self.ty.data.zero_()
        self.tz.data.zero_()

class Homo:
    def __init__(self, device='cuda:0'):
        self.H = torch.eye(4, device=device) # 4x4 homogeneous transformation matrix
        self.device = device

    def apply_differentials(self, dx, dy, dz, dr, dtx, dty, dtz):
        # should run this with torch.no_grad()
        R = torch.eye(4, device=self.device)
        R[:3, :3] = mat(dtx, dty, dtz)

        S = torch.eye(4, device=self.device)
        S[[0, 1, 2], [0, 1, 2]] = dr.exp()

        T = torch.eye(4, device=self.device)
        T[0, 3] = dx.item()
        T[1, 3] = dy.item()
        T[2, 3] = dz.item()
        
        self.H = T @ S @ R @ self.H

def mat(dtx, dty, dtz, device='cuda:0'):
    device = dtx.device if isinstance(dtx, torch.Tensor) else device

    if isinstance(dtx, float):
        dtx = torch.tensor([dtx], dtype=torch.float32, device=device)
    if isinstance(dty, float):
        dty = torch.tensor([dty], dtype=torch.float32, device=device)
    if isinstance(dtz, float):
        dtz = torch.tensor([dtz], dtype=torch.float32, device=device)

    # rotate along z-axis
    Rtx = torch.stack([
        torch.concat([torch.cos(dtx), -torch.sin(dtx), torch.tensor([0]).to(device)], axis=0),
        torch.concat([torch.sin(dtx), torch.cos(dtx), torch.tensor([0]).to(device)], axis=0),
        torch.concat([torch.tensor([0]).to(device), torch.tensor([0]).to(device), torch.tensor([1]).to(device)], axis=0)
    ], axis=0) 
    # rotate along x-axis
    Rty = torch.stack([
        torch.concat([torch.tensor([1]).to(device), torch.tensor([0]).to(device), torch.tensor([0]).to(device)], axis=0),
        torch.concat([torch.tensor([0]).to(device), torch.cos(dty), -torch.sin(dty)], axis=0),
        torch.concat([torch.tensor([0]).to(device), torch.sin(dty), torch.cos(dty)], axis=0)
    ], axis=0)
    # rotate along y-axis
    Rtz = torch.stack([
        torch.concat([torch.cos(dtz), torch.tensor([0]).to(device), torch.sin(dtz)], axis=0),
        torch.concat([torch.tensor([0]).to(device), torch.tensor([1]).to(device), torch.tensor([0]).to(device)], axis=0),
        torch.concat([-torch.sin(dtz), torch.tensor([0]).to(device), torch.cos(dtz)], axis=0)
    ], axis=0)

    return Rtz @ Rty @ Rtx

def DP_PCR(X, Y, epochs=2, faster=True, device='cuda:0'):
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)

    g = learn_pdf(Y, model='KDE', device=device)
    f = learn_pdf(X, model='KDE', device=device)

    return register(X, Y, f, g, epochs=epochs, batch_size=8_000, faster=faster, device=device)

def register(X_og, Y_og, f, g, epochs=3000, batch_size=20_000, faster=True, device="cuda:0"):
    """
    X and Y are shape (N, 3) and (M, 3)
    faster: True for faster running and more memory usage.  False for slower running and less memory usage
    """   
    debug_interval = 20
    debug = False

    _og_pos = Y_og.mean(axis=0)     
    _og_scale = torch.linalg.norm(Y_og - _og_pos, axis=1).mean()

    d = Differential(device)
    H = Homo(device)

    grads = []
    chamfers = []

    # actually using one optimizer would suffice
    optimizer1 = torch.optim.SGD([d.x, d.y, d.z, d.r], lr=9e-4)
    optimizer2 = torch.optim.SGD([d.tx, d.ty, d.tz], lr=9e-5)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs // 1.2, eta_min=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs // 1.2, eta_min=1e-5)

    X = X_og.clone()
    Y = Y_og.clone()

    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        optimizer1.zero_grad()
        optimizer2.zero_grad()


        # sample from x and y, can increase batch_size as long as GPU memory isn't busted 
        indices_x = torch.randperm(X.shape[0])[:batch_size] # shape (batch_size, latent_size)
        indices_y = torch.randperm(Y.shape[0])[:batch_size] # shape (batch_size, latent_size)
        
        # handle indexing
        x_batch = X[indices_x]
        y_batch = Y[indices_y]

        # scalar values, only used during debugging
        loss = 0.0
        
        bigD = (d.r.exp() * torch.eye(3, device=device)) @ mat(d.tx, d.ty, d.tz)
        _T = torch.concat([d.x, d.y, d.z]).unsqueeze(0) # shape (1, 3)

        if (not faster): # longer running time but much less GPU memory usage
            # does backprop for each loss component separately to greatly reduce GPU ram usage
            # trans losses
            temp = -torch.mean( g.log_prob((bigD @ (x_batch + _T).T).T) )
            if ~(torch.isnan(temp) | torch.isinf(temp)):
                temp.backward()
                loss += temp.item()
            
            temp = -torch.mean( f.log_prob(
                    (torch.linalg.inv(bigD) @ (y_batch).T).T - _T
            ))
            if ~(torch.isnan(temp) | torch.isinf(temp)):
                temp.backward()
                loss += temp.item()
            
        else: # for big GPU, can do backprop all at once
            temp = \
                - torch.mean( g.log_prob((bigD @ (x_batch + _T).T).T) ) \
                - torch.mean( f.log_prob(
                    (torch.linalg.inv(bigD) @ (y_batch).T).T - _T
                ))
            
            if ~(torch.isnan(temp) | torch.isinf(temp)):
                temp.backward()
            # below only used for visualizing loss changes during training
            loss += temp.item()

        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()

        # grads.append(
        #     np.concat([
        #         _.detach().cpu() for _ in (d.x, d.y, d.z, d.r, d.tx, d.ty, d.tz)
        #     ])
        # )

        # update point cloud X based on differentials
        with torch.no_grad():
            H.apply_differentials(d.x, d.y, d.z, d.r, d.tx, d.ty, d.tz)

            X = (H.H[:3, :3] @ X_og.T).T + H.H[:3, 3]
            Y = (H.H.inverse()[:3, :3] @ Y_og.T).T + H.H.inverse()[:3, 3]
            d.zero_() # zero out differentials        

            chamfers.append(chamfer_distance(X.unsqueeze(0), Y.unsqueeze(0))[0].item())


            if debug and ((epoch % debug_interval == 0) or (epoch == epochs - 1)):
                # set torch to only print 3 decimal places and no scientific notation
                torch.set_printoptions(precision=3, sci_mode=False)
                print('---------------------')
                print(f'Epoch {epoch}, Loss: {loss:.3f}')
                print(' | '.join(
                    [f"{name} {param.grad.item():.3f}" for (name, param) in zip(['dx', 'dy', 'dz', 'dr', 'dtx', 'dty', 'dtz'], [d.x, d.y, d.z, d.r, d.tx, d.ty, d.tz])]
                ))
                # print(' | '.join(
                #     [f"{name} {param.grad.item():.3f}" for (name, param) in zip(['dx', 'dy', 'dz', 'dr', 'dtx'], [d.x, d.y, d.z, d.r, d.tx])]
                # ))
                print(f"current position: {X.mean(axis=0)}, current scale: {torch.linalg.norm(X-X.mean(axis=0), axis=1).mean():.3f}")
                print(f"GT position     : {_og_pos}, GT      scale: {_og_scale:.3f}")
                
                print(H.H)
                print('---------------------')

                # plot registered result
                visualize(
                    [((H.H[:3, :3] @ X_og.T).T + H.H[:3, 3]).detach().cpu().numpy(), Y_og.detach().cpu().numpy()],
                    ['blue', 'red'],
                    show = False,
                    save = f"figs/E{epoch}_coupled.png",
                    marker_size = 0.7
                )

            

        # Update loss on progress bar
        progress_bar.set_postfix({'Loss': f"{loss:.2f}"})


    if debug:
        with torch.no_grad():
            # plot registered result
            visualize(
                [((H.H[:3, :3] @ X_og.T).T + H.H[:3, 3]).detach().cpu().numpy(), Y_og.detach().cpu().numpy()],
                ['blue', 'red'],
                show = False,
                save = f"figs/E{epochs}_coupled.png",
                marker_size = 0.7
            )


    # np.savetxt("grads_coupled.txt", np.array(grads))
    np.savetxt("del_coupled.txt", np.array(chamfers))
    return H.H.cpu().numpy()

