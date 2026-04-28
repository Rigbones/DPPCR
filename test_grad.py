# python libraries
import numpy as np
import os

# my functions
from utils import axis_angle_to_matrix
from format_conversions import ply_to_np
import trainer_coupled
import trainer

coupled_list = []
decoupled_list = []

filenames = list(os.listdir("datasets/smol/"))
filenames.sort()

device = 'cuda:1'
mode = 'rotation' # 'rotation', 'scale', or 'translation'
half = 'second' # 'first' or 'second'
filenames = filenames[:450] if half == 'first' else filenames[450:]

for name in filenames:
    seed = int(name.split('_')[-1].split('.')[0]) # eg get 0681 from sofa_0681.ply
    rng = np.random.default_rng(seed)
    Y = ply_to_np("datasets/smol/" + name)

    ### Rotation ###
    if (mode == 'rotation'):
        R1 = axis_angle_to_matrix([1, 0, 0], np.radians(rng.uniform(0, 40))) # (3, 3)
        R2 = axis_angle_to_matrix([0, 1, 0], np.radians(rng.uniform(0, 40))) # (3, 3)
        R3 = axis_angle_to_matrix([0, 0, 1], np.radians(rng.uniform(0, 40))) # (3, 3)
        X = ((R1 @ R2 @ R3) @ Y.T).T
    ### Scale ###
    elif (mode == 'scale'):
        S = np.array([rng.uniform(0.3, 0.7) if rng.random() < 0.5 else rng.uniform(1.3, 1.7)]) # (1,)
        X = S * X
    ### Translation ###
    elif (mode == 'translation'):
        T = rng.uniform(-2.0, 2.0, size=3) # (3,)
        X = Y + T
    else:
        raise ValueError(f"Invalid mode {mode}")

    print(f"Running {name} on {device}, mode: {mode}, half: {half}")
    trainer_coupled.DP_PCR(X, Y, epochs=300, device=device)
    trainer.DP_PCR(X, Y, epochs=300, device=device)

    coupled_list.append(np.loadtxt('grads_coupled.txt'))
    decoupled_list.append(np.loadtxt('grads_decoupled.txt'))

    # shape (908, 300, 7)
    np.save(f"{mode}_coupled_{half}.npy", np.stack(coupled_list, axis=0))
    np.save(f"{mode}_decoupled_{half}.npy", np.stack(decoupled_list, axis=0))



