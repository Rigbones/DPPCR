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

def print_memory_used():
    print(f"{torch.cuda.device_memory_used(0) / 1024 / 1024 / 1024:.2f} GB")

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

def DP_PCR(X, Y, device='cuda:0'):
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)

    g1 = learn_pdf(Y[:, 0:1], model='KDE', device=device)
    g2 = learn_pdf(Y[:, 1:2], model='KDE', device=device)
    g3 = learn_pdf(Y[:, 2:3], model='KDE', device=device)
    g4 = learn_pdf(torch.linalg.norm(Y-Y.mean(axis=0), axis=1, keepdim=True), model='KDE', device=device)
    g5 = learn_pdf((Y-Y.mean(axis=0))[:, 0:2], model='KDE', device=device)
    g6 = learn_pdf((Y-Y.mean(axis=0))[:, 1:3], model='KDE', device=device)
    g7 = learn_pdf((Y-Y.mean(axis=0))[:, [0, 2]], model='KDE', device=device)

    f1 = learn_pdf(X[:, 0:1], model='KDE', device=device)
    f2 = learn_pdf(X[:, 1:2], model='KDE', device=device)
    f3 = learn_pdf(X[:, 2:3], model='KDE', device=device)
    f4 = learn_pdf(torch.linalg.norm(X-X.mean(axis=0), axis=1, keepdim=True), model='KDE', device=device)
    f5 = learn_pdf((X-X.mean(axis=0))[:, 0:2], model='KDE', device=device)
    f6 = learn_pdf((X-X.mean(axis=0))[:, 1:3], model='KDE', device=device)
    f7 = learn_pdf((X-X.mean(axis=0))[:, [0, 2]], model='KDE', device=device)

    return register(X, Y, f1, f2, f3, f4, f5, f6, f7, g1, g2, g3, g4, g5, g6, g7, epochs=500, batch_size=7000, device=device)

def register(X_og, Y_og, f1, f2, f3, f4, f5, f6, f7, g1, g2, g3, g4, g5, g6, g7, epochs=3000, batch_size=20_000, device="cuda:0"):
    """X and Y are shape (N, 3) and (M, 3)"""   
    debug_interval = 50

    _og_pos = Y_og.mean(axis=0)     
    _og_scale = torch.linalg.norm(Y_og - _og_pos, axis=1).mean()

    d = Differential(device)
    H = Homo(device)

    grads = []

    # actually using one optimizer would suffice
    optimizer1 = torch.optim.SGD([d.x, d.y, d.z, d.r], lr=9e-3)
    optimizer2 = torch.optim.SGD([d.tx, d.ty, d.tz], lr=9e-4)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs // 1.2, eta_min=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs // 1.2, eta_min=1e-5)

    X = X_og.clone()
    Y = Y_og.clone()

    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # handle X
        x1, x2, x3 = X.unbind(axis=-1) # Each has shape (N,)
        x4 = torch.linalg.norm(X - X.mean(axis=0), axis=-1)

        x123_centered = torch.stack([x1, x2, x3], axis=-1)
        x123_centered = x123_centered - x123_centered.mean(axis=0)

        # handle Y
        y1, y2, y3 = Y.unbind(axis=-1) # Each has shape (M,)
        y4 = torch.linalg.norm(Y - Y.mean(axis=0), axis=-1)

        y123_centered = torch.stack([y1, y2, y3], axis=-1)
        y123_centered = y123_centered - y123_centered.mean(axis=0)

        # sample from x and y, can increase batch_size as long as GPU memory isn't busted 
        indices_x = torch.randperm(x1.shape[0])[:batch_size] # shape (batch_size, latent_size)
        indices_y = torch.randperm(y1.shape[0])[:batch_size] # shape (batch_size, latent_size)
        
        x1_batch = x1[indices_x] # shape (batch_size,)
        x2_batch = x2[indices_x] 
        x3_batch = x3[indices_x]
        x4_batch = x4[indices_x]
        x123_centered_batch = x123_centered[indices_x]
        y1_batch = y1[indices_y] # shape (batch_size,)
        y2_batch = y2[indices_y]
        y3_batch = y3[indices_y]
        y4_batch = y4[indices_y]
        y123_centered_batch = y123_centered[indices_y]

        loss_trans = \
            - torch.mean( g1.log_prob(x1_batch.unsqueeze(1) + d.x) ) \
            - torch.mean( g2.log_prob(x2_batch.unsqueeze(1) + d.y) ) \
            - torch.mean( g3.log_prob(x3_batch.unsqueeze(1) + d.z) ) \
            - torch.mean( f1.log_prob(y1_batch.unsqueeze(1) - d.x) ) \
            - torch.mean( f2.log_prob(y2_batch.unsqueeze(1) - d.y) ) \
            - torch.mean( f3.log_prob(y3_batch.unsqueeze(1) - d.z) )
        
        loss_scale = \
            - torch.mean( g4.log_prob((d.r).exp() * x4_batch.unsqueeze(1)) ) \
            - torch.mean( f4.log_prob((-d.r).exp() * y4_batch.unsqueeze(1)) )

        loss_rot = \
            - torch.mean( g5.log_prob((mat(d.tx, 0.0, 0.0, device) @ x123_centered_batch.T).T[:, 0:2]) ) \
            - torch.mean( g6.log_prob((mat(0.0, d.ty, 0.0, device) @ x123_centered_batch.T).T[:, 1:3]) ) \
            - torch.mean( g7.log_prob((mat(0.0, 0.0, d.tz, device) @ x123_centered_batch.T).T[:, [0, 2]]) ) \
            - torch.mean( f5.log_prob((mat(d.tx, 0.0, 0.0, device).T @ y123_centered_batch.T).T[:, 0:2]) ) \
            - torch.mean( f6.log_prob((mat(0.0, d.ty, 0.0, device).T @ y123_centered_batch.T).T[:, 1:3]) ) \
            - torch.mean( f7.log_prob((mat(0.0, 0.0, d.tz, device).T @ y123_centered_batch.T).T[:, [0, 2]]) )
        
        loss = loss_trans + loss_scale + loss_rot

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

            # grads.append(
            #     np.concat([
            #         _.grad.detach().cpu() for _ in (d.x, d.y, d.z, d.r, d.tx, d.ty, d.tz)
            #     ])
            # )

        # update point cloud X based on differentials
        with torch.no_grad():
            H.apply_differentials(d.x, d.y, d.z, d.r, d.tx, d.ty, d.tz)

            X = (H.H[:3, :3] @ X_og.T).T + H.H[:3, 3]
            Y = (H.H.inverse()[:3, :3] @ Y_og.T).T + H.H.inverse()[:3, 3]
            d.zero_() # zero out differentials        

            # if (epoch % debug_interval == 0) or (epoch == epochs - 1):
            #     # set torch to only print 3 decimal places and no scientific notation
            #     torch.set_printoptions(precision=3, sci_mode=False)
            #     print('---------------------')
            #     print(f'Epoch {epoch}, Loss: {loss.item():.3f} = {loss_trans.item():.3f} + {loss_scale.item():.3f} + {loss_rot.item():.3f}')
            #     print(' | '.join(
            #         [f"{name} {param.grad.item():.3f}" for (name, param) in zip(['dx', 'dy', 'dz', 'dr', 'dtx', 'dty', 'dtz'], [d.x, d.y, d.z, d.r, d.tx, d.ty, d.tz])]
            #     ))
            #     # print(' | '.join(
            #     #     [f"{name} {param.grad.item():.3f}" for (name, param) in zip(['dx', 'dy', 'dz', 'dr', 'dtx'], [d.x, d.y, d.z, d.r, d.tx])]
            #     # ))
            #     print(f"current position: {X.mean(axis=0)}, current scale: {torch.linalg.norm(X-X.mean(axis=0), axis=1).mean():.3f}")
            #     print(f"GT position     : {_og_pos}, GT      scale: {_og_scale:.3f}")
                
            #     print(H.H)
            #     print('---------------------')

            #     # save plots as 3 figures
            #     fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, _), (ax9, ax10, ax11, _)) = plt.subplots(3, 4, figsize=(24, 10), layout='tight')
        
            #     # x, y, z, r
            #     _ = ax1.hist(X.detach().cpu().numpy()[:, 0], bins=100, alpha=0.5, density=True, label='pred')
            #     _ = ax1.hist(Y_og.detach().cpu().numpy()[:, 0], bins=100, alpha=0.5, density=True, label='GT')
            #     ax1.legend()
            #     ax1.set_title("X axis")

            #     _ = ax2.hist(X.detach().cpu().numpy()[:, 1], bins=100, alpha=0.5, density=True, label='pred')
            #     _ = ax2.hist(Y_og.detach().cpu().numpy()[:, 1], bins=100, alpha=0.5, density=True, label='GT')
            #     ax2.legend()
            #     ax2.set_title("Y axis")

            #     _ = ax3.hist(X.detach().cpu().numpy()[:, 2], bins=100, alpha=0.5, density=True, label='pred')
            #     _ = ax3.hist(Y_og.detach().cpu().numpy()[:, 2], bins=100, alpha=0.5, density=True, label='GT')
            #     ax3.legend()
            #     ax3.set_title("Z axis")

            #     _ = ax4.hist((X-X.mean(axis=0)).norm(dim=1).detach().cpu().numpy(), bins=100, alpha=0.5, density=True, label='pred')
            #     _ = ax4.hist((Y_og-Y_og.mean(axis=0)).norm(dim=1).detach().cpu().numpy(), bins=100, alpha=0.5, density=True, label='GT')
            #     ax4.legend()
            #     ax4.set_title("Norm")

            #     # xy, yz, zr
            #     _ = ax5.hist2d((X-X.mean(axis=0))[:, 0].detach().cpu().numpy(), (X-X.mean(axis=0))[:, 1].detach().cpu().numpy(), bins=100)
            #     _ = ax9.hist2d((Y_og-Y_og.mean(axis=0))[:, 0].detach().cpu().numpy(), (Y_og-Y_og.mean(axis=0))[:, 1].detach().cpu().numpy(), bins=100)
            #     ax5.set_title("XY Projection (pred)")
            #     ax9.set_title("XY Projection ( GT )")

            #     _ = ax6.hist2d((X-X.mean(axis=0))[:, 1].detach().cpu().numpy(), (X-X.mean(axis=0))[:, 2].detach().cpu().numpy(), bins=100)
            #     _ = ax10.hist2d((Y_og-Y_og.mean(axis=0))[:, 1].detach().cpu().numpy(), (Y_og-Y_og.mean(axis=0))[:, 2].detach().cpu().numpy(), bins=100)
            #     ax6.set_title("YZ Projection (pred)")
            #     ax10.set_title("YZ Projection ( GT )")

            #     _ = ax7.hist2d((X-X.mean(axis=0))[:, 0].detach().cpu().numpy(), (X-X.mean(axis=0))[:, 2].detach().cpu().numpy(), bins=100)
            #     _ = ax11.hist2d((Y_og-Y_og.mean(axis=0))[:, 0].detach().cpu().numpy(), (Y_og-Y_og.mean(axis=0))[:, 2].detach().cpu().numpy(), bins=100)
            #     ax7.set_title("XZ Projection (pred)")
            #     ax11.set_title("XZ Projection ( GT )")

            #     fig.savefig(f"figs/E{epoch}.png")
            #     plt.close('all')

                # if (input() == 'q'):
                #     exit(0)    

        # Update loss on progress bar
        progress_bar.set_postfix({'Loss': f"{loss.item():.2f}"})

    # with torch.no_grad():
    #     # plot original pcd
    #     visualize(
    #         [X_og.detach().cpu().numpy(), Y_og.detach().cpu().numpy()],
    #         ['blue', 'red'],
    #         show = False,
    #         save = f"figs/E{epochs}_original.png",
    #         marker_size = 0.7
    #     )

    #     # plot registered result
    #     visualize(
    #         [((H.H[:3, :3] @ X_og.T).T + H.H[:3, 3]).detach().cpu().numpy(), Y_og.detach().cpu().numpy()],
    #         ['blue', 'red'],
    #         show = False,
    #         save = f"figs/E{epochs}_result.png",
    #         marker_size = 0.7
    #     )

    return H.H.cpu().numpy()

