
from utils import axis_angle_to_matrix, jitter, add_noise

if __name__ == "__main__":
    X = np.load(f"your_pcd.npy") # shape (N, 3)
    Y = copy.deepcopy(X)

    Y = Y - Y.mean(axis=0) # shape (N, 3)
    X = X - X.mean(axis=0)

    rng = np.random.default_rng()
    # get ground truth transformation
    GT_R1 = axis_angle_to_matrix([1, 0, 0], np.radians(rng.uniform(0, 25)))
    GT_R2 = axis_angle_to_matrix([0, 1, 0], np.radians(rng.uniform(0, 25)))
    GT_R3 = axis_angle_to_matrix([0, 0, 1], np.radians(rng.uniform(0, 25)))
    GT_S = np.array([rng.uniform(0.5, 0.7) if rng.random() < 0.5 else rng.uniform(1.3, 1.5)])
    GT_T = rng.uniform(0.5, 0.5, size=3)

    # apply transformations to X
    GT = np.eye(4)
    GT[:3, :3] = GT_R3 @ GT_R2 @ GT_R1
    GT[:3, 3] = GT_T
    X = GT_S * (GT[:3, :3] @ X.T).T + GT[:3, 3]

    np.save('X.npy', X)
    np.save('Y.npy', Y)

    X = add_noise(jitter(X))
    print(GT)

    if (not np.isclose(np.linalg.det(GT_R1), 1.0)) or (not np.isclose(np.linalg.det(GT_R2), 1.0)) or (not np.isclose(np.linalg.det(GT_R3), 1.0)):
        raise Exception("One of the rotation matrices have non-zero determinants")

    start_time = time.perf_counter() # Record the start time
    DP_PCR(X, Y, device='cuda:0')
    end_time = time.perf_counter()   # Record the end time
    
    # write elapsed time to a file
    print(end_time - start_time)