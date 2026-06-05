import numpy as np

def fleiss_kappa(ratings: np.ndarray) -> float:
    n_subjects, n_raters = ratings.shape
    categories = np.unique(ratings[~np.isnan(ratings)])
    n_categories = len(categories)
    freq_matrix = np.zeros((n_subjects, n_categories))
    for i, cat in enumerate(categories):
        freq_matrix[:, i] = np.sum(ratings == cat, axis=1)
    P_i = (np.sum(freq_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)
    p_j = np.sum(freq_matrix, axis=0) / (n_subjects * n_raters)
    P_e = np.sum(p_j ** 2)
    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)

def bootstrap_ci(data: np.ndarray, n_bootstrap: int=10000, ci: float=0.95, seed: int | None=None) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point_estimate = float(np.mean(data))
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
    alpha = 1 - ci
    ci_lower = float(np.percentile(bootstrap_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
    return (point_estimate, ci_lower, ci_upper)
