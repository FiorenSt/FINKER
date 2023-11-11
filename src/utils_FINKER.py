### LOCAL LINEAR REGRESSION AND LOCAL CONSTANT REGRESSION
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import iqr
from numpy.linalg import inv, LinAlgError, qr, solve, matrix_rank
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
import joblib
from joblib import Parallel, delayed



# Integrating the first five functions into the EfficientStats class

class FINKER:
    def __init__(self, **kwargs):
        """
        Initializes the FINKER class with user-defined parameters.

        Parameters:
        -----------
        **kwargs : dict
            Keyword arguments for setting various parameters.
        """
        # Default parameters
        self.params = {
            'use_grid': True,
            'alpha': 0.0618,
            'bandwidth_method': 'custom',
            # Add other default parameters here
        }

        # Models for bootstrap
        self.glm_model_path = 'glm_poly_model.pkl'  # Example attribute
        self.poly_transform_path = 'poly_transform.pkl'  # Example attribute

        # Update with user-provided parameters
        self.params.update(kwargs)

    def set_parameter(self, key, value):
        """
        Set or update a parameter.
        """
        self.params[key] = value


    def gaussian_kernel(self, u, l_squared):
        """
        Computes the Gaussian kernel.

        Parameters:
        -----------
        u : array_like
            The input array.
        l_squared : float
            The squared bandwidth parameter.

        Returns:
        --------
        np.ndarray
            The computed Gaussian kernel values.
        """
        return np.exp(-u ** 2 / (2 * l_squared))


    def silverman_bw(self, data):
        """
        Calculates bandwidth based on Silverman's rule.

        Parameters:
        -----------
        data : array_like
            The input data.

        Returns:
        --------
        float
            The calculated bandwidth.
        """
        n = len(data)
        std_dev = np.std(data)
        IQR = iqr(data) / 1.34
        A = np.min([std_dev, IQR])  # Pass the values in a list or array
        return 0.9 * A * n ** (-1 / 5)

    def custom_bw(self, data, alpha=None):
        """
        Calculates bandwidth based on a custom rule for locally periodic data.

        Parameters:
        -----------
        data : array_like
            The input data.
        alpha : float, optional
            A scaling factor for the bandwidth.

        Returns:
        --------
        float
            The calculated bandwidth.
        """
        n = len(data)
        # Combine the standard deviation of the data and the period of the signal
        return alpha * n ** (-1 / 5) if alpha is not None else None

    def adaptive_bw(self, data, k=None, metric='euclidean'):
        """
        Calculates the average distance to k-nearest neighbors for each point in data.

        Parameters:
        -----------
        data : array_like
            The input data.
        k : int, optional
            The number of nearest neighbors.
        metric : str, optional
            The distance metric to use.

        Returns:
        --------
        tuple
            The average bandwidths and the NearestNeighbors instance.
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

    def kernel_regression_local_constant(self, x, x_observed, y_observed, uncertainties, l_squared, kernel_type='gaussian'):
        """
        Performs kernel regression with a local constant approach.

        Parameters:
        -----------
        x : array_like
            The points at which to evaluate the regression.
        x_observed : array_like
            The observed x-values.
        y_observed : array_like
            The observed y-values.
        uncertainties : array_like or None
            The uncertainties associated with y-values, or None.
        l_squared : float
            The squared bandwidth.
        f : float
            Frequency for phase folding.
        kernel_type : str, optional
            Type of kernel to use ('gaussian', 'locally_periodic', 'periodic').

        Returns:
        --------
        float
            The regression result at point x.
        """
        if kernel_type == 'gaussian':
            kernel_weights = self.gaussian_kernel(x - x_observed, l_squared)
        else:
            raise ValueError("Invalid kernel_type. Choose among 'gaussian'.")

        if uncertainties is not None:
            uncertainty_weights = 1 / (uncertainties ** 2)
            combined_weights = kernel_weights * uncertainty_weights
        else:
            combined_weights = kernel_weights

        epsilon = 1e-10  # A small constant to avoid division by zero
        return np.sum(y_observed * combined_weights) / (np.sum(combined_weights) + epsilon)


    def kernel_regression_local_linear(self, x, x_observed, y_observed, uncertainties, l_squared, kernel_type='gaussian'):
        """
        Performs kernel regression with a local linear approach.

        Parameters:
        -----------
        x : array_like
            The points at which to evaluate the regression.
        x_observed : array_like
            The observed x-values.
        y_observed : array_like
            The observed y-values.
        uncertainties : array_like or None
            The uncertainties associated with y-values, or None.
        l_squared : float
            The squared bandwidth.
        f : float
            Frequency for phase folding.
        kernel_type : str, optional
            Type of kernel to use ('gaussian', 'locally_periodic', 'periodic').

        Returns:
        --------
        float
            The regression result at point x.
        """
        n = len(x_observed)

        if kernel_type == 'gaussian':
            kernel_weights = self.gaussian_kernel(x - x_observed, l_squared)
        else:
            raise ValueError("Invalid kernel_type. Choose among 'gaussian', 'locally_periodic', or 'periodic'.")

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
            # Fallback to QR decomposition in case of singular matrix
            ridge_term = 1e-6  # Small ridge term to ensure invertibility
            A = X.T @ W @ X
            B = X.T @ W @ y_observed

            # Ensure full rank for matrix A
            if matrix_rank(A) < min(A.shape):
                A += np.eye(A.shape[0]) * ridge_term

            # QR decomposition
            Q, R = qr(A)

            # Ensure full rank for matrix R
            if matrix_rank(R) < min(R.shape):
                R += np.eye(R.shape[0]) * ridge_term

            # Solve R*beta = Q^T * B
            beta_hat = solve(R, Q.T @ B)

        y_hat = beta_hat[0] + beta_hat[1] * x
        return y_hat

    def closest_point_mapping(self, points, reference_points):
        """
        Maps each point to the closest point in a set of reference points.

        Parameters:
        -----------
        points : array_like
            The points to be mapped.
        reference_points : array_like
            The reference points.

        Returns:
        --------
        array_like
            The closest reference point for each input point.
        """
        distances = cdist(points.reshape(-1, 1), reference_points.reshape(-1, 1))
        min_indices = np.argmin(distances, axis=1)
        return reference_points[min_indices]


    def nonparametric_kernel_regression(self, t_observed, y_observed, freq, l=None, alpha=None,
                                        uncertainties=None, grid_size=300,
                                        show_plot=False, kernel_type='gaussian',
                                        regression_type='local_constant', bandwidth_method='silverman',
                                        adaptive_neighbours=None, use_grid=True):
        """
        Performs nonparametric kernel regression.

        Parameters:
        -----------
        t_observed : array_like
            The observed time values.
        y_observed : array_like
            The observed y-values.
        freq : float
            Frequency for phase folding.
        l : float, optional
            Bandwidth for the kernel.
        alpha : float, optional
            Scaling factor for custom bandwidth.
        uncertainties : array_like or None, optional
            The uncertainties associated with y-values, or None.
        grid_size : int, optional
            Size of the grid for evaluation.
        show_plot : bool, optional
            If True, show the plot of the results.
        kernel_type : str, optional
            Type of kernel to use.
        regression_type : str, optional
            Type of regression to perform.
        bandwidth_method : str, optional
            Method for bandwidth selection.
        adaptive_neighbours : int, optional
            Number of neighbours for adaptive bandwidth.
        use_grid : bool, optional
            If True, use grid for evaluation, else use observed points.

        Returns:
        --------
        tuple
            The regression results and other relevant information.
        """

        phase = (t_observed * freq) % 1

        if l is None:
            if bandwidth_method == 'silverman':
                l = self.silverman_bw(phase)
            elif bandwidth_method == 'custom':
                l = self.custom_bw(phase, alpha)
            elif bandwidth_method == 'adaptive':
                # Initialization logic for adaptive bandwidth if needed
                l = None
            else:
                raise ValueError(
                    "Invalid bandwidth_method. Choose among 'silverman', 'custom', or 'adaptive'.")

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
            uncertainties_extended = np.concatenate(
                [uncertainties_mirrored_upper, uncertainties_sorted, uncertainties_mirrored_lower])
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
            regression_function = self.kernel_regression_local_constant
        elif regression_type == 'local_linear':
            regression_function = self.kernel_regression_local_linear
        else:
            raise ValueError("Invalid regression_type. Choose either 'local_constant' or 'local_linear'.")

        # Initialization for adaptive bandwidth
        adaptive_bandwidths = None
        if bandwidth_method == 'adaptive':
            if adaptive_neighbours is None:
                adaptive_neighbours = int(np.log(len(phase_extended)))
            adaptive_bandwidths, nbrs = self.adaptive_bw(phase_extended.reshape(-1, 1), k=adaptive_neighbours)
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
            y_estimate = regression_function(x, phase_extended, y_extended, uncertainties_extended, l_squared,
                                             kernel_type)
            y_estimates.append(y_estimate)

        # Post-processing and plotting
        if use_grid:
            mask_in_interval = (x_eval >= 0) & (x_eval <= 1)
            x_eval_in_interval = x_eval[mask_in_interval]
            y_estimates_in_interval = np.array(y_estimates)[mask_in_interval]
        else:
            x_eval_in_interval = x_eval
            y_estimates_in_interval = np.array(y_estimates)

        closest_x_grid = self.closest_point_mapping(phase_sorted, x_eval_in_interval)
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
                             markersize=5, label='Observed with Uncertainty', zorder=0)
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



    def parallel_nonparametric_kernel_regression(self, t_observed, y_observed, freq_list, data_type='periodic',
                                                 n_jobs=-2, verbose=0, n_bootstrap=1000, **kwargs):
        """
        Apply nonparametric_kernel_regression for a list of frequencies in parallel,
        find the best frequency, and estimate the uncertainties.

        Parameters:
        -----------
        t_observed : array_like
            The observed time values of the light curve.
        y_observed : array_like
            The observed y-values (e.g., magnitude) of the light curve.
        freq_list : array_like
            List of frequencies for phase folding.
        data_type : str, optional
            Type of astronomical data ('general', 'eclipse', 'binary').
        n_jobs : int, optional
            The number of jobs to run in parallel.
        n_bootstrap : int
            The number of bootstrap samples for uncertainty estimation.
        **kwargs : dict
            Keyword arguments to pass to nonparametric_kernel_regression.

        Returns:
        --------
        tuple
            Best frequency, estimated uncertainty, significance status, and result dictionary.
        """
        self.params.update(kwargs)

        def task_for_each_frequency(freq):
            result = self.nonparametric_kernel_regression(t_observed, y_observed, freq,
                                                          kernel_type=self.params['kernel_type'],
                                                          regression_type=self.params['regression_type'],
                                                          bandwidth_method=self.params['bandwidth_method'],
                                                          alpha=self.params['alpha'],
                                                          use_grid=self.params['use_grid'])
            return result[4] # Returning squared_residual and residuals

        # Parallel processing
        with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
            broad_results = parallel(delayed(task_for_each_frequency)(freq) for freq in freq_list)
        broad_result_dict = dict(zip(freq_list, broad_results))

        initial_best_freq = freq_list[np.argmin(broad_results)]

        # Significance checks
        if data_type in ['periodic']:
            significance_status, significant_freq = self.check_frequency_significance(initial_best_freq,
                                                                                      broad_results, freq_list)
        else:
            significance_status = 'not_applicable'
            significant_freq = initial_best_freq

        # Tight frequency range search
        search_width = 0.01
        tight_freq_range = np.linspace(significant_freq * (1 - search_width), significant_freq * (1 + search_width),
                                       10000)
        tight_results = parallel(delayed(task_for_each_frequency)(freq) for freq in tight_freq_range)
        tight_result_dict = dict(zip(tight_freq_range, tight_results))

        final_best_freq = min(tight_result_dict, key=tight_result_dict.get)
        final_best_residual = tight_result_dict[final_best_freq]
        combined_result_dict = {**broad_result_dict, **tight_result_dict}

        # Bootstrap uncertainty estimation with residuals at optimal frequency
        estimated_uncertainty = self.bootstrap_uncertainty_estimation(
            t_observed, y_observed, kwargs.get('uncertainties'), final_best_freq, n_bootstrap, len(t_observed), n_jobs,
            residual=final_best_residual
        )

        return final_best_freq, estimated_uncertainty, significance_status, combined_result_dict, final_best_residual


    def check_frequency_significance(self, freq, objective_values, freq_range, search_width=0.001, **kwargs):
        """
        Check the significance of a frequency and its multiples for the highest significant match.

        Parameters:
        -----------
        freq : float
            The frequency to check.
        objective_values : array_like
            Objective values for each frequency in the range.
        freq_range : array_like
            The range of frequencies to consider.
        search_width : float, optional
            The width for searching around the significant frequency.
        **kwargs : dict, optional
            Additional class-level parameters.

        Returns:
        --------
        tuple
            The significance status and the best frequency within the range.
        """

        # Update class-level parameters if any
        self.params.update(kwargs)

        significance_threshold = np.percentile(objective_values, .1)

        # Checking multiples of the frequency
        multiples = [2, 3, 4, 5]
        significant_multiples = []
        for multiple in multiples:
            test_freq = freq * multiple
            test_freq_idx = np.abs(freq_range - test_freq).argmin()
            test_freq_obj_val = objective_values[test_freq_idx]

            if test_freq_obj_val < significance_threshold:
                significant_multiples.append((multiple, test_freq, test_freq_idx))

        # Finding the highest significant multiple
        if significant_multiples:
            highest_significant_multiple = max(significant_multiples, key=lambda x: x[0])
            ratio, test_freq, test_freq_idx = highest_significant_multiple

            # Search range around the highest significant frequency
            lower_bound = test_freq - search_width * test_freq
            upper_bound = test_freq + search_width * test_freq

            # Find the best frequency within the search range
            search_indices = [i for i, f in enumerate(freq_range) if lower_bound <= f <= upper_bound]
            if search_indices:
                best_in_range_idx = search_indices[np.argmin([objective_values[i] for i in search_indices])]
                return f'multiple_{ratio}', freq_range[best_in_range_idx]
            else:
                return f'multiple_{ratio}', freq_range[test_freq_idx]

        # Checking subharmonics if no significant multiples are found
        subharmonics = [0.5, 1/3, 1/4, 1/5]
        for subharmonic in subharmonics:
            test_freq = freq * subharmonic
            test_freq_idx = np.abs(freq_range - test_freq).argmin()
            test_freq_obj_val = objective_values[test_freq_idx]

            if test_freq_obj_val < significance_threshold:
                break  # Subharmonic is significant, but not the main frequency

        # Return the original frequency if no significant multiples are found
        return 'best', freq


    def bootstrap_uncertainty_estimation(self, t_observed, y_observed, uncertainties, best_freq, n_bootstrap, sample_size, n_jobs, residual, **kwargs):
        """
        Perform bootstrap sampling and kernel regression to estimate uncertainties.

        Parameters:
        -----------
        t_observed : array_like
            The observed time values of the light curve.
        y_observed : array_like
            The observed y-values (e.g., magnitude) of the light curve.
        uncertainties : array_like or None
            The uncertainties associated with y-values, or None.
        best_freq : float
            The best frequency determined from the previous analysis.
        n_bootstrap : int
            The number of bootstrap samples to generate.
        sample_size : int
            The size of the sample.
        n_jobs : int
            The number of jobs to run in parallel.

        Returns:
        --------
        float
            The estimated true uncertainty.
        """
        # Update class-level parameters with any additional arguments provided
        self.params.update(kwargs)

        # Capture necessary parameters
        current_params = self.params

        def bootstrap_task(_):
            # Accessing method parameters dynamically
            if uncertainties is not None:
                t_resampled, y_resampled, uncertainties_resampled = resample(t_observed, y_observed, uncertainties)
                _, _, _, _, squared_residual, _ = self.nonparametric_kernel_regression(
                    t_resampled, y_resampled, best_freq, uncertainties=uncertainties_resampled,
                    use_grid=current_params['use_grid'],
                    bandwidth_method=current_params['bandwidth_method'],
                    alpha=current_params['alpha']
                )
            else:
                t_resampled, y_resampled = resample(t_observed, y_observed)
                _, _, _, _, squared_residual, _ = self.nonparametric_kernel_regression(
                    t_resampled, y_resampled, best_freq,
                    use_grid=current_params['use_grid'],
                    bandwidth_method=current_params['bandwidth_method'],
                    alpha=current_params['alpha']
                )
            return squared_residual

        # Bootstrap sampling using joblib's Parallel
        with Parallel(n_jobs=n_jobs) as parallel:
            bootstrap_results = parallel(delayed(bootstrap_task)(_) for _ in range(n_bootstrap))

        bootstrap_variability = np.std(bootstrap_results)/residual

        # Load the saved model and PolynomialFeatures object
        glm_poly = joblib.load(self.glm_model_path)
        poly = joblib.load(self.poly_transform_path)

        # Predict the estimated uncertainty
        x2 = np.log10(sample_size)
        x1 = bootstrap_variability
        new_X = np.array([[x1, x2]])
        new_X_transformed = poly.transform(new_X)

        estimated_uncertainty = glm_poly.predict(new_X_transformed)[0]
        return estimated_uncertainty


    def process_multiband_data(self, multiband_data, freq_list, **kwargs):
        """
        Process multiband observational data, combine and normalize residuals.

        Parameters:
        -----------
        multiband_data : dict
            Dictionary containing observational data for each band.
            Format: {'band_name': (t_observed, y_observed, uncertainties), ...}
        freq_list : array_like
            List of frequencies for phase folding.
        **kwargs : dict
            Additional arguments for regression methods.

        Returns:
        --------
        dict
            Combined results from all bands and normalized combined residuals.
        """
        combined_results = {}
        total_sample_size = 0
        combined_scaled_residuals = 0

        for band, (t_observed, y_observed, uncertainties) in multiband_data.items():
            # Update parameters for current band
            self.params.update(kwargs.get(band, {}))

            # Perform parallel regression for the current band
            best_freq, estimated_uncertainty, significance_status, result_dict, final_best_residual = \
                self.parallel_nonparametric_kernel_regression(
                    t_observed, y_observed, freq_list, uncertainties=uncertainties, **self.params
                )

            # Store results for the current band
            combined_results[band] = {
                'best_frequency': best_freq,
                'estimated_uncertainty': estimated_uncertainty,
                'significance_status': significance_status,
                'result_dict': result_dict,
                'final_best_residuals': final_best_residual
            }

            # Extract, scale, and sum squared residuals
            band_sample_size = len(t_observed)
            total_sample_size += band_sample_size
            band_residuals = result_dict['final_best_residuals']  # Extracting residuals from the result dictionary
            combined_scaled_residuals += band_residuals / band_sample_size

        # Normalize the combined residuals
        if total_sample_size > 0:
            combined_scaled_residuals /= total_sample_size

        combined_results['combined_scaled_residuals'] = combined_scaled_residuals

        return combined_results

