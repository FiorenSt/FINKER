### LOCAL LINEAR REGRESSION AND LOCAL CONSTANT REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import cdist
from scipy.stats import iqr
from numpy.linalg import inv, LinAlgError, qr, solve, matrix_rank
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity


## KERNELS

import numpy as np

# Gaussian Kernel
def gaussian_kernel(u, l_squared):
    return np.exp(-u ** 2 / (2 * l_squared))

# Complex Periodic Kernel
def periodic_kernel(u, l, p):
    periodic_part = np.exp(-2 * np.sin(np.pi * np.abs(u) / p)**2 )#/ l**2)
    se_part = np.exp(-u**2 / (2 * l**2))
    return periodic_part * se_part

# Locally Periodic Kernel (Complex)
def locally_periodic_kernel(u, l, p):
    periodic_part = np.exp(-2 * np.sin(np.pi * np.abs(u) / p)**2 / l**2)
    se_part = np.exp(-u**2 / (2 * l**2))
    return periodic_part * se_part


## BANDWIDTH

# Function to calculate bandwidth based on Silverman's rule
def silverman_bw(data):
    n = len(data)
    std_dev = np.std(data)
    IQR = iqr(data)/1.34
    A = np.min([std_dev, IQR])  # Pass the values in a list or array
    return 0.9 * A * n ** (-1 / 5)


# Function to calculate bandwidth based on custom rule for locally periodic data
def custom_bw(data, alpha=None):
    n = len(data)
    # Combine the standard deviation of the data and the period of the signal
    return alpha * n ** (-1 / 5)


def heuristic_median_bw(data):
    """
    Calculates the heuristic median bandwidth for the given 1D data array.

    Parameters:
    - data: np.array, the 1D data for which to calculate the bandwidth

    Returns:
    - bandwidth: float, the heuristic median bandwidth
    """
    # Calculate pairwise distances for 1D array
    pairwise_distances = np.abs(data[:, np.newaxis] - data[np.newaxis, :])

    # Get the lower triangle (excluding diagonal) to avoid duplicate distances
    lower_triangle_indices = np.tril_indices_from(pairwise_distances, k=-1)
    lower_triangle_distances = pairwise_distances[lower_triangle_indices]

    # Calculate the median distance
    bandwidth = np.median(lower_triangle_distances)

    return bandwidth


def adaptive_bw(data, k=None, metric='euclidean'):
    """
    Calculates the average distance to k-nearest neighbors for each point in data.
    """
    if k is None:
        k = 10  # Default value, can be set to any desired number

    # Initialize NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric=metric).fit(data)

    # Get the distances and indices of k+1 nearest neighbors for each point
    distances, _ = nbrs.kneighbors(data)

    # Calculate the average distance to k nearest neighbors, skipping the 0-th index
    avg_bandwidths = np.mean(distances[:, 1:], axis=1)

    return avg_bandwidths, nbrs



############################################ Regression Functions ######################################################

def kernel_regression_local_constant(x, x_observed, y_observed, uncertainties, l_squared, f, kernel_type='gaussian'):

    if kernel_type == 'locally_periodic':
        kernel_weights = locally_periodic_kernel(x - x_observed, l_squared, 1/f)
    elif kernel_type == 'gaussian':
        kernel_weights = gaussian_kernel(x - x_observed, l_squared)
    elif kernel_type == 'periodic':
        kernel_weights = periodic_kernel(x - x_observed, l_squared, 1/f)
    else:
        raise ValueError("Invalid kernel_type. Choose among 'locally_periodic', 'gaussian', or 'periodic'.")

    if uncertainties is not None:
        uncertainty_weights = 1 / (uncertainties ** 2)
        combined_weights = kernel_weights * uncertainty_weights
    else:
        combined_weights = kernel_weights

    epsilon = 1e-10  # A small constant
    return np.sum(y_observed * combined_weights) / (np.sum(combined_weights) + epsilon)


def kernel_regression_local_linear(x, x_observed, y_observed, uncertainties, l_squared, f,
                                   kernel_type='gaussian'):
    n = len(x_observed)

    if kernel_type == 'locally_periodic':
        kernel_weights = locally_periodic_kernel(x - x_observed, l_squared, 1 / f)
    elif kernel_type == 'gaussian':
        kernel_weights = gaussian_kernel(x - x_observed, l_squared)
    elif kernel_type == 'periodic':
        kernel_weights = periodic_kernel(x - x_observed, l_squared, 1 / f)
    else:
        raise ValueError("Invalid kernel_type. Choose among 'locally_periodic', 'gaussian', or 'periodic'.")

    if uncertainties is not None:
        uncertainty_weights = 1 / (uncertainties ** 2)
        kernel_weights *= uncertainty_weights

    # Prepare design matrix
    X = np.column_stack([np.ones(n), x_observed])

    # Compute weighted least squares estimates
    W = np.diag(kernel_weights)

    try:
        # Try direct inversion first
        beta_hat = inv(X.T @ W @ X) @ X.T @ W @ y_observed
    except LinAlgError:
        print("Matrix is singular or nearly singular. Switching to QR decomposition.")

        ridge_term = 1e-6  # Small ridge term to ensure invertibility

        # If direct inversion fails, use QR decomposition
        A = X.T @ W @ X
        B = X.T @ W @ y_observed

        # Check if A is full rank
        if matrix_rank(A) < min(A.shape):
            # print("Matrix is rank deficient. Adding ridge term.")
            A += np.eye(A.shape[0]) * ridge_term

        # QR decomposition
        Q, R = qr(A)

        # Check if R is full rank
        if matrix_rank(R) < min(R.shape):
            # print("Matrix is rank deficient even after QR. Adding ridge term.")
            R += np.eye(R.shape[0]) * ridge_term

        # Solve R*beta = Q^T * B
        beta_hat = solve(R, Q.T @ B)

        # The local linear regression estimate at x
    y_hat = beta_hat[0] + beta_hat[1] * x

    return y_hat


## INTERPOLATION TO EVALUATE AT OBSERVED POINTS
def closest_point_mapping(points, reference_points):
    distances = cdist(points.reshape(-1, 1), reference_points.reshape(-1, 1))
    min_indices = np.argmin(distances, axis=1)
    return reference_points[min_indices]



def nonparametric_kernel_regression(t_observed, y_observed, freq, l=None, alpha=None,
                                    uncertainties=None, grid_size=300,
                                    show_plot=False, kernel_type='gaussian',
                                    regression_type='local_constant', bandwidth_method='silverman',
                                    adaptive_neighbours=None, use_grid=True):
    # Your existing preprocessing code
    phase = (t_observed * freq) % 1

    if l is None:
        if bandwidth_method == 'silverman':
            l = silverman_bw(phase)
        elif bandwidth_method == 'heuristic':
            l = heuristic_median_bw(phase)
        elif bandwidth_method == 'custom':
            l = custom_bw(phase, alpha)
        elif bandwidth_method == 'adaptive':
            # Initialization logic for adaptive bandwidth if needed
            l = None
        else:
            raise ValueError(
                "Invalid bandwidth_method. Choose among 'silverman', 'heuristic', 'custom', or 'adaptive'.")

    # Determine number of points to mirror based on the dataset size
    n_mirror = int(1.0 / np.sqrt(len(t_observed)) * len(phase))

    # Sort the phase and corresponding y-values
    sort_indices = np.argsort(phase)
    phase_sorted = phase[sort_indices]
    y_sorted = y_observed[sort_indices]

    # Create mirrored points for the upper boundary (to be placed before 0)
    phase_mirrored_upper = phase_sorted[-n_mirror:] - 1
    y_mirrored_upper = y_sorted[-n_mirror:]

    # Create mirrored points for the lower boundary (to be placed past 1)
    phase_mirrored_lower = phase_sorted[:n_mirror] + 1
    y_mirrored_lower = y_sorted[:n_mirror]

    # Mirror uncertainties if they are provided
    if uncertainties is not None:
        uncertainties_sorted = uncertainties[sort_indices]
        uncertainties_mirrored_upper = uncertainties_sorted[-n_mirror:]
        uncertainties_mirrored_lower = uncertainties_sorted[:n_mirror]
        uncertainties_extended = np.concatenate([uncertainties_mirrored_upper, uncertainties_sorted, uncertainties_mirrored_lower])
    else:
        uncertainties_extended = None

    # Combine the original and mirrored points without modulus operation
    phase_extended = np.concatenate([phase_mirrored_upper, phase_sorted, phase_mirrored_lower])
    y_extended = np.concatenate([y_mirrored_upper, y_sorted, y_mirrored_lower])

    # Choose the points for evaluation based on the use_grid option
    if use_grid:
        grid_start = phase_extended.min() - (1.0 / grid_size)  # Slightly before the minimum
        grid_end = phase_extended.max() + (1.0 / grid_size)  # Slightly after the maximum
        x_eval = np.linspace(grid_start, grid_end, grid_size)
    else:
        x_eval = phase_sorted  # Directly use observed points for evaluation

    # Function mapping based on regression_type
    if regression_type == 'local_constant':
        regression_function = kernel_regression_local_constant
    elif regression_type == 'local_linear':
        regression_function = kernel_regression_local_linear
    else:
        raise ValueError("Invalid regression_type. Choose either 'local_constant' or 'local_linear'.")

    # Initialization for adaptive bandwidth
    adaptive_bandwidths = None
    if bandwidth_method == 'adaptive':
        if adaptive_neighbours is None:
            adaptive_neighbours = int(np.log(len(phase_extended)))
        adaptive_bandwidths, nbrs = adaptive_bw(phase_extended.reshape(-1, 1), k=adaptive_neighbours)
    else:
        l_squared = l ** 2

    # Main loop for kernel regression
    y_estimates = []
    for x in x_eval:
        if bandwidth_method == 'adaptive':
            # Find k nearest neighbors and their distances
            distances, _ = nbrs.kneighbors(np.array([[x]]))
            # Use the average distance as the bandwidth for this specific x
            l = np.mean(distances)
            l_squared = l ** 2
        y_estimate = regression_function(x, phase_extended, y_extended, uncertainties_extended, l_squared, freq, kernel_type)
        y_estimates.append(y_estimate)

    # Post-processing and plotting
    if use_grid:
        mask_in_interval = (x_eval >= 0) & (x_eval <= 1)
        x_eval_in_interval = x_eval[mask_in_interval]
        y_estimates_in_interval = np.array(y_estimates)[mask_in_interval]
    else:
        x_eval_in_interval = x_eval
        y_estimates_in_interval = np.array(y_estimates)

    closest_x_grid = closest_point_mapping(phase_sorted, x_eval_in_interval)
    mapping_dict = dict(zip(x_eval_in_interval, y_estimates_in_interval))
    y_smoothed_original = np.array([mapping_dict[x] for x in closest_x_grid])

    if uncertainties is not None:
        # Compute the weighted residuals
        weighted_residuals = (y_smoothed_original - y_observed[sort_indices]) / uncertainties[sort_indices]

        # Calculate the squared residual
        squared_residual = np.sum(weighted_residuals ** 2) / len(t_observed)

    else:
        # Compute the unweighted residuals
        residuals = y_smoothed_original - y_observed[sort_indices]

        # Calculate the squared residual
        squared_residual = np.sum(residuals ** 2) / len(t_observed)


    if show_plot:
        plt.figure(figsize=(10, 6))
        if uncertainties is not None:
            plt.errorbar(phase_sorted, y_observed[sort_indices], yerr=uncertainties[sort_indices], fmt='.',
                         markersize=5, label='Observed with Uncertainty',zorder=0)
        else:
            plt.scatter(phase_sorted, y_observed[sort_indices], s=5, label='Observed')
        plt.plot(x_eval_in_interval, y_estimates_in_interval, 'r-', label='Smoothed', linewidth=2)
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        title_str = f"Phase-folded Light Curve with {kernel_type.replace('_', ' ').title()} Kernel"
        title_str += f" using {regression_type.replace('_', ' ').title()} Regression"
        plt.title(title_str)
        plt.legend()
        plt.gca().invert_yaxis()  # Invert the y-axis to match astronomical magnitude convention
        plt.show()

    # Return the standard deviation of residuals along with other values
    if bandwidth_method == 'adaptive':
        return phase_sorted, y_smoothed_original, x_eval_in_interval, y_estimates_in_interval, squared_residual, adaptive_bandwidths
    else:
        return phase_sorted, y_smoothed_original, x_eval_in_interval, y_estimates_in_interval, squared_residual, l





def parallel_nonparametric_kernel_regression(t_observed, y_observed, freq_list, data_type='periodic',
                                             n_jobs=-2, verbose=0, n_bootstrap=1000, **kwargs):
    """
    Apply nonparametric_kernel_regression for a list of frequencies in parallel, find the best frequency,
    and estimate the uncertainties.

    Parameters:
        t_observed (array-like): The observed time values of the light curve.
        y_observed (array-like): The observed y-values (e.g., magnitude) of the light curve.
        freq_list (array-like): List of frequencies for phase folding.
        data_type (str): Type of astronomical data ('general', 'eclipse', 'binary').
        n_jobs (int, optional): The number of jobs to run in parallel.
        n_bootstrap (int): The number of bootstrap samples for uncertainty estimation.
        regression_params (dict): Parameters of the regression model for predicting true error.
        **kwargs: Keyword arguments to pass to nonparametric_kernel_regression.

    Returns:
        best_freq (float): The frequency with the lowest objective metric.
        estimated_uncertainty (float): The estimated true uncertainty.
        significance_status (str): The significance status of the best frequency.
        result_dict (dict): A dictionary where keys are frequencies and values are the specified metric(s).
    """

    def task_for_each_frequency(freq):
        # Include uncertainties if provided, otherwise pass None
        result = nonparametric_kernel_regression(t_observed, y_observed, freq, **kwargs)
        return result[4]  # Only returning squared_residual

    with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
        broad_results = parallel(delayed(task_for_each_frequency)(freq) for freq in freq_list)
    broad_result_dict = dict(zip(freq_list, broad_results))

    initial_best_freq = freq_list[np.argmin(broad_results)]

    # Perform significance checks based on data type
    if data_type in ['periodic']:
        significance_status, significant_freq = check_frequency_significance(initial_best_freq, broad_results, freq_list)
    else:
        significance_status = 'not_applicable'
        significant_freq = initial_best_freq

    search_width = 0.01
    tight_freq_range = np.linspace(significant_freq * (1 - search_width), significant_freq * (1 + search_width), 1000)
    tight_results = parallel(delayed(task_for_each_frequency)(freq) for freq in tight_freq_range)
    tight_result_dict = dict(zip(tight_freq_range, tight_results))

    final_best_freq = min(tight_result_dict, key=tight_result_dict.get)
    combined_result_dict = {**broad_result_dict, **tight_result_dict}

    # Bootstrap Uncertainty Estimation
    estimated_uncertainty = bootstrap_uncertainty_estimation(
        t_observed, y_observed, kwargs.get('uncertainties'), final_best_freq, n_bootstrap, len(t_observed), n_jobs
    )

    return final_best_freq, estimated_uncertainty, combined_result_dict





## SIGNIFICANCE CHECKS

def check_frequency_significance(freq, objective_values, freq_range, search_width=0.001):
    """
    Check the significance of a frequency and its multiples for the highest significant match.
    Subharmonics are used for significance check but the main frequency is the highest significant multiple.
    """
    significance_threshold = np.percentile(objective_values, .1)

    # Define the range for multiples
    multiples = [2, 3, 4, 5]  # Adjust as needed

    significant_multiples = []
    for multiple in multiples:
        test_freq = freq * multiple
        test_freq_idx = np.abs(freq_range - test_freq).argmin()
        test_freq_obj_val = objective_values[test_freq_idx]

        if test_freq_obj_val < significance_threshold:
            significant_multiples.append((multiple, test_freq, test_freq_idx))

    # Find the highest significant multiple
    if significant_multiples:
        highest_significant_multiple = max(significant_multiples, key=lambda x: x[0])
        ratio, test_freq, test_freq_idx = highest_significant_multiple
        print(f'Multiple {ratio}x is significant. Searching around this frequency for the best candidate.')

        # Define search range around the highest significant frequency
        lower_bound = test_freq - search_width * test_freq
        upper_bound = test_freq + search_width * test_freq

        # Find the best frequency within the search range
        search_indices = [i for i, f in enumerate(freq_range) if lower_bound <= f <= upper_bound]
        if search_indices:
            best_in_range_idx = search_indices[np.argmin([objective_values[i] for i in search_indices])]
            return f'multiple_{ratio}', freq_range[best_in_range_idx]
        else:
            return f'multiple_{ratio}', freq_range[test_freq_idx]

    # If no significant multiples are found, consider subharmonics for significance check
    subharmonics = [0.5, 1/3, 1/4, 1/5]  # Adjust as needed
    for subharmonic in subharmonics:
        test_freq = freq * subharmonic
        test_freq_idx = np.abs(freq_range - test_freq).argmin()
        test_freq_obj_val = objective_values[test_freq_idx]

        if test_freq_obj_val < significance_threshold:
            print(f'Subharmonic {subharmonic}x is significant, but not the main frequency.')
            break

    # Returning the original frequency if no significant multiples are found
    print('No significant multiples found. Considering the initial frequency.')
    return 'best', freq





from sklearn.utils import resample


from sklearn.utils import resample

def bootstrap_uncertainty_estimation(t_observed, y_observed, uncertainties, best_freq, n_bootstrap, sample_size, n_jobs=-2):
    """
    Perform bootstrap sampling and kernel regression to estimate uncertainties.

    Parameters:
        t_observed (array-like): The observed time values of the light curve.
        y_observed (array-like): The observed y-values (e.g., magnitude) of the light curve.
        uncertainties (array-like or None): The uncertainties associated with y-values, or None.
        best_freq (float): The best frequency determined from the previous analysis.
        n_bootstrap (int): The number of bootstrap samples to generate.
        sample_size (int): The size of the sample.
        n_jobs (int): The number of jobs to run in parallel.

    Returns:
        estimated_uncertainty (float): The estimated true uncertainty.
    """

    def bootstrap_task(_):
        if uncertainties is not None:
            t_resampled, y_resampled, uncertainties_resampled = resample(t_observed, y_observed, uncertainties)
            _, _, _, _, squared_residual, _ = nonparametric_kernel_regression(
                t_resampled, y_resampled, best_freq, uncertainties=uncertainties_resampled, use_grid=True
            )
        else:
            t_resampled, y_resampled = resample(t_observed, y_observed)
            _, _, _, _, squared_residual, _ = nonparametric_kernel_regression(
                t_resampled, y_resampled, best_freq, use_grid=True
            )

        return squared_residual

    with Parallel(n_jobs=n_jobs) as parallel:
        bootstrap_results = parallel(delayed(bootstrap_task)(_) for _ in range(n_bootstrap))

    bootstrap_variability = np.std(bootstrap_results)

    # Hypothetical regression parameters
    bootstrap_coefficient = 1.5
    sample_size_coefficient = -0.002
    intercept = 0.05

    # Predicting the true error
    estimated_uncertainty = (bootstrap_variability * bootstrap_coefficient) + (sample_size * sample_size_coefficient) + intercept

    return estimated_uncertainty































### GENERATING SYNTHETIC LIGHT CURVES

import pandas as pd
from astropy.table import Table

def sort_on_x(x,y):

    zipp = list(zip(x,y))
    zipp.sort(key=lambda x:x[0])
    x,y = list(zip(*zipp))

    return np.array(x), np.array(y)


def get_mag_stddev_relation(upper=13, lower=20,degree=5):

    dat = Table.read('DATA/16010_ML1_16010_q_20210428_red_cat.fits',format='fits')
    df = dat.to_pandas()
    df = df.loc[ ( (df.MAG_OPT>upper) & (df.MAG_OPT<lower) ) ]
    x = np.array(df.MAG_OPT.tolist())
    y = np.sqrt(1.)*np.array(df.MAGERR_OPT.tolist())
    xx, yy = sort_on_x(x,y)
    dy = pd.Series(yy)
    yq = np.array(dy.rolling(500).quantile(0.25).tolist())

    idx = np.isfinite(yq)
    yq = yq[idx]
    xx = xx[idx]

    fx = np.polynomial.Polynomial.fit(xx, yq, degree, domain=[0,1], window=[0,1])

    ## !! returns function !!
    return fx




def generate_synthetic_light_curve(n_points=500, time=10, freq_primary=1,
                                    amplitude_primary=1, freq_secondary=1,
                                    amplitude_secondary=0, eclipse_depth=0,
                                    baseline_magnitude=17.0, noise_function = None,
                                    n_repeats=10, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    if noise_function is None:
        noise_function = get_mag_stddev_relation()
    noise_std = noise_function(baseline_magnitude)


    # Time vector
    t_observed = np.sort(np.random.uniform(0, time, n_points))
    # Generate primary frequency component
    y_primary = - amplitude_primary * np.sin(2 * np.pi * freq_primary * t_observed)

    # Generate secondary frequency component
    y_secondary = - amplitude_secondary * np.sin(2 * np.pi * freq_secondary * t_observed)
    # Generate eclipse (simplified)
    y_eclipse = eclipse_depth * np.sin(2 * np.pi * freq_secondary * t_observed) ** 30
    # Combine to get the complex light curve in magnitudes
    y_magnitude = baseline_magnitude + y_primary + y_secondary + y_eclipse
    # Simulate repeated observations
    repeated_observations = np.zeros((n_repeats, n_points))
    for i in range(n_repeats):
        uncertainties = np.random.normal(0, noise_std, n_points)
        repeated_observations[i, :] = y_magnitude + uncertainties

    # Calculate the mean and standard deviation of the repeated observations
    y_magnitude_observed = np.mean(repeated_observations, axis=0)
    sigma = np.std(repeated_observations, axis=0)
    return t_observed, y_magnitude_observed, sigma











####  ENTROPY FUNCTIONS

def conditional_entropy(phase, y, M=10, N=10):
    phase_bins = np.linspace(0, 1, M + 1)
    mag_bins = np.linspace(min(y), max(y), N + 1)
    hist2d, _, _ = np.histogram2d(phase, y, bins=(phase_bins, mag_bins))
    p_xi_yj = hist2d / np.sum(hist2d)
    hist1d, _ = np.histogram(phase, bins=phase_bins)
    p_xi = hist1d / np.sum(hist1d)
    H = -np.sum(p_xi_yj * np.log((p_xi_yj + 1e-10) / (p_xi[:, None] + 1e-10)))
    return H

def perform_conditional_entropy_phase_folding(t, y_observed, freq, M=10, N=10, show_plot=False):
    """
    Parameters:
    - t (array-like): The time values of the light curve
    - y_observed (array-like): The observed y-values of the light curve
    - freq (float): The frequency for phase folding
    - M, N (int, optional): Number of bins for phase and magnitude. Default is 10.
    - show_plot (bool, optional): Whether to show a plot. Default is False.

    Returns:
    - phase (array-like): Phase-folded time values
    - H (float): The conditional entropy of the original phase-folded points
    """

    # Phase fold the light curve
    phase = (t * freq) % 1

    # Calculate conditional entropy
    H = conditional_entropy(phase, y_observed, M=M, N=N)

    # If show_plot is True, display the plot
    if show_plot:
        sort_indices = np.argsort(phase)
        phase_sorted = phase[sort_indices]
        y_observed_sorted = y_observed[sort_indices]

        plt.figure(figsize=(10, 6))
        plt.scatter(phase_sorted, y_observed_sorted, s=5, label='Observed')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.title('Phase-folded Light Curve')
        plt.legend()
        plt.show()

    return phase, H


def objective_function_entropy(freq, t, y_observed, N=10, M=10):
    _, H = perform_conditional_entropy_phase_folding(t, y_observed, freq, N, M)
    return H  # Return conditional entropy instead of squared_residual


from joblib import Parallel, delayed

def parallel_conditional_entropy(t_observed, y_observed, freq_list, M=10, N=10, n_jobs=-2, verbose=0, search_width=0.01):
    """
    Calculate conditional entropy for a list of frequencies in parallel.
    Perform a focused search around the best frequency found.
    """

    def task_for_each_frequency(freq):
        _, H = perform_conditional_entropy_phase_folding(t_observed, y_observed, freq, M=M, N=N)
        return H

    # Broad frequency search
    with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
        entropy_results = parallel(delayed(task_for_each_frequency)(freq) for freq in freq_list)

    broad_result_dict = dict(zip(freq_list, entropy_results))
    best_freq = min(broad_result_dict, key=broad_result_dict.get)

    # Tighter grid search around the best frequency
    tight_freq_range = np.linspace(best_freq * (1 - search_width), best_freq * (1 + search_width), 1000)
    with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
        tight_entropy_results = parallel(delayed(task_for_each_frequency)(freq) for freq in tight_freq_range)

    tight_result_dict = dict(zip(tight_freq_range, tight_entropy_results))
    best_tight_freq = min(tight_result_dict, key=tight_result_dict.get)
    best_tight_entropy = tight_result_dict[best_tight_freq]

    # Combine results from both searches
    combined_result_dict = {**broad_result_dict, **tight_result_dict}

    return best_tight_freq, best_tight_entropy, combined_result_dict

