import numpy as np


def pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """
    X: (n_samples, d)
    return: (n_samples, n_samples), D[i,j] = ||x_i - x_j||^2
    """
    x2 = np.sum(X * X, axis=1, keepdims=True)
    D = x2 + x2.T - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    return D


def build_knn_laplacian(
    X: np.ndarray,
    n_neighbors: int = 5,
    weight_mode: str = "binary",
    heat_kernel_t: float = 1.0,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    Use KNN to construct the Laplacian matrix L = D - S
    Returns L : (n_samples, n_samples)
    """
    n = X.shape[0]
    if not (1 <= n_neighbors < n):
        raise ValueError("n_neighbors must be in [1, n_samples-1].")

    D2 = pairwise_squared_distances(X)
    nn_idx = np.argsort(D2, axis=1)[:, 1:n_neighbors + 1]

    S = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in nn_idx[i]:
            if weight_mode == "binary":
                w = 1.0
            elif weight_mode == "heat":
                w = np.exp(-D2[i, j] / max(heat_kernel_t, 1e-12))
            else:
                raise ValueError("weight_mode must be 'binary' or 'heat'.")
            S[i, j] = w

    if symmetrize:
        S = np.maximum(S, S.T)

    degree = np.sum(S, axis=1)
    L = np.diag(degree) - S
    return L


def lapgod_greedy(
    X: np.ndarray,
    n_select: int,
    alpha: float = 1e-2,
    beta: float = 1e-3,
    n_neighbors: int = 2,
    weight_mode: str = "binary",
    heat_kernel_t: float = 1.0,
    return_objective_history: bool = True,
):
    """
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, d)
    n_select : int
    alpha, beta : float
    n_neighbors : int
    weight_mode : str
    heat_kernel_t : float
    return_objective_history : bool

    Returns
    -------
    selected_indices : list[int]
    objective_history : list[float]
    L : np.ndarray
    """
    n_samples, d = X.shape
    if not (1 <= n_select <= n_samples):
        raise ValueError("n_select must be in [1, n_samples].")

    # KNN graph Laplacian
    L = build_knn_laplacian(
        X,
        n_neighbors=n_neighbors,
        weight_mode=weight_mode,
        heat_kernel_t=heat_kernel_t,
        symmetrize=True,
    )

    # convert to column-major for faster access in inner loop
    X_col = X.T  # (d, n_samples)

    # initial regularizer P_0 = alpha X L X^T + beta I
    # P0 = alpha X L X^T + beta I
    regularizer = alpha * (X_col @ L @ X_col.T) + beta * np.eye(d, dtype=np.float64)

    P_inv = np.linalg.inv(regularizer)

    # current V_k V_k^T，initalized to 0
    VVt = np.zeros((d, d), dtype=np.float64)

    selected = []
    selected_mask = np.zeros(n_samples, dtype=bool)
    objective_history = []

    for step in range(n_select):
        best_idx = -1
        best_obj = np.inf
        best_candidate_P_inv = None
        best_candidate_VVt = None

        for j in range(n_samples):
            if selected_mask[j]:
                continue

            v = X[j].reshape(d, 1)  

            # A = (P_k + vv^T)^(-1)
            Pv = P_inv @ v
            denom = 1.0 + float(v.T @ Pv)
            A = P_inv - (Pv @ Pv.T) / denom

            # B = V_k V_k^T + vv^T
            B = VVt + (v @ v.T)

            # x_i^T A B A x_i
            # compute M = A B A first
            M = A @ B @ A

            # compute in batch
            # X 是 (n_samples, d)
            # vals[i] = x_i^T M x_i
            MXt = M @ X_col              # (d, n_samples)
            vals = np.sum(X_col * MXt, axis=0)   # (n_samples,)
            obj = float(np.max(vals))

            if obj < best_obj:
                best_obj = obj
                best_idx = j
                best_candidate_P_inv = A
                best_candidate_VVt = B

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True
        P_inv = best_candidate_P_inv
        VVt = best_candidate_VVt
        objective_history.append(best_obj)

    if return_objective_history:
        return selected, objective_history, L
    return selected, L