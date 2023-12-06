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
            'show_bootstrap_histogram': True
        }

        # Update with user-provided parameters
        self.params.update(kwargs)

    def set_parameter(self, key, value):
        """
        Set or update a parameter.
        """
        self.params[key] = value

    def gaussian_kernel(self, u, l_squared):
        """
        Computes the Gaussian kernel with normalization.

        Parameters:
        -----------
        u : array_like
            The input array.
        l_squared : float
            The squared bandwidth parameter.

        Returns:
        --------
        np.ndarray
            The computed Gaussian kernel values with normalization.
        """
        return (1 / np.sqrt(2 * np.pi * l_squared)) * np.exp(-u ** 2 / (2 * l_squared))


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
        return alpha * (n ** (-1 / 5)) if alpha is not None else None

    def adaptive_bw(self, data, metric='euclidean'):
        """
        Calculates adaptive bandwidth based on the distance to the k-th nearest neighbor.

        Parameters:
        -----------
        data : array_like
            The input data.

        Returns:
        --------
        float
            The calculated adaptive bandwidth.
        """
        n = len(data)
        #k = round(0.538 * n ** (4/5))
        k = round(np.log(n))


        # Calculate the distance to the k-th nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric).fit(data)
        distances, _ = nbrs.kneighbors(data)

        # Bandwidth for each point is the distance to its k-th nearest neighbor
        bandwidths = np.maximum(np.mean(distances[:, 1:], axis=1), 1e-6)
        return bandwidths



    def kernel_regression_local_constant(self, x, x_observed, y_observed, l_squared, kernel_type='gaussian'):
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

        # A small constant to avoid division by zero
        return np.sum(y_observed * kernel_weights) / (np.sum(kernel_weights))


    def kernel_regression_local_linear(self, x, x_observed, y_observed, l_squared, kernel_type='gaussian'):
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


    def nonparametric_kernel_regression(self, t_observed, y_observed, uncertainties, freq, l=None, alpha=None,
                                        grid_size=300,
                                        show_plot=False, kernel_type='gaussian',
                                        regression_type='local_constant', bandwidth_method='custom',
                                        use_grid=False):
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
            adaptive_bandwidths = self.adaptive_bw(phase_extended.reshape(-1, 1))
        else:
            l_squared = l ** 2

        # Main loop for kernel regression
        y_estimates = []
        for i, x in enumerate(x_eval):
            if bandwidth_method == 'adaptive':
                # Use the adaptive bandwidth for the current point
                l_squared = adaptive_bandwidths[i] ** 2
            # Perform regression with the selected bandwidth
            y_estimate = regression_function(x=x, x_observed=phase_extended, y_observed=y_extended, l_squared=l_squared,
                                             kernel_type=kernel_type)
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


        # Compute the unweighted residuals
        normalized_uncertainties = (uncertainties - uncertainties.mean()) / uncertainties.std()
        normalized_uncertainties = normalized_uncertainties + np.abs(normalized_uncertainties.min()) + 1

        # Compute the weighted residuals
        residuals = (y_smoothed_original - y_observed[sort_indices]) / normalized_uncertainties[sort_indices]
        #residuals = (y_smoothed_original - y_observed[sort_indices]) / uncertainties[sort_indices]

        # Calculate the squared residual
        squared_residual = np.sum(residuals ** 2) / len(t_observed)

        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.errorbar(phase_sorted, y_observed[sort_indices], yerr=uncertainties[sort_indices], fmt='.',
                         markersize=5, label='Observed with Uncertainty', zorder=0)

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

    def parallel_nonparametric_kernel_regression(self, t_observed, y_observed, uncertainties, freq_list, use_grid=None,
                                                 n_jobs=-2, verbose=0, n_bootstrap=1000,
                                                 tight_check_points=1000, search_width=0.001,
                                                 estimate_uncertainties=False, bootstrap_points=100, bootstrap_width=0.005, save_bootstrap_freq = False,
                                                 **kwargs):
        """
        Apply nonparametric_kernel_regression for a list of frequencies in parallel,
        find the best frequency, and optionally estimate the uncertainties.

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
        enable_tight_check : bool, optional
            Whether to perform an additional tight frequency check.
        tight_check_points : int, optional
            Number of points to use in the tight frequency check.
        search_width : float, optional
            The width of the search range in the tight check.
        estimate_uncertainties : bool, optional
            Whether to estimate uncertainties using bootstrap.
        **kwargs : dict
            Keyword arguments to pass to nonparametric_kernel_regression.

        Returns:
        --------
        tuple
            Best frequency, estimated uncertainty (if calculated), significance status, and result dictionary.
        """

        self.params.update(kwargs)

        if use_grid is None:
            use_grid = len(t_observed) > 300
            grid_size = 300 if use_grid else None

        if use_grid:
            print('Using a 300 points grid.')

        self.set_parameter('use_grid',use_grid)


        def task_for_each_frequency(freq):
            result = self.nonparametric_kernel_regression(t_observed=t_observed, y_observed=y_observed,
                                                          uncertainties=uncertainties,
                                                          freq=freq,
                                                          kernel_type=self.params['kernel_type'],
                                                          regression_type=self.params['regression_type'],
                                                          bandwidth_method=self.params['bandwidth_method'],
                                                          alpha=self.params['alpha'],
                                                          use_grid=use_grid, grid_size=grid_size)
            return result[4]  # Returning squared_residual and residuals

        # Parallel processing
        with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
            broad_results = parallel(delayed(task_for_each_frequency)(freq) for freq in freq_list)

        initial_best_freq = freq_list[np.argmin(broad_results)]

        # Significance checks
        significance_results = self.check_frequency_significance(initial_best_freq, broad_results, freq_list)
        significant_frequencies = [freq for _, freq in significance_results]

        # Initialize dictionary to hold all results
        combined_result_dict = dict(zip(freq_list, broad_results))

        for freq in significant_frequencies:
            # Tight frequency range search for each significant frequency
            tight_freq_range = np.linspace(freq * (1 - search_width), freq * (1 + search_width), tight_check_points)
            tight_results = parallel(delayed(task_for_each_frequency)(f) for f in tight_freq_range)
            combined_result_dict.update(dict(zip(tight_freq_range, tight_results)))

        # Find the frequency with the overall smallest residual
        final_best_freq = min(combined_result_dict, key=combined_result_dict.get)


        estimated_uncertainty = None
        if estimate_uncertainties:
            estimated_uncertainty = self.bootstrap_uncertainty_estimation(
                t_observed, y_observed, uncertainties, final_best_freq, n_bootstrap,
                n_jobs,bootstrap_points,bootstrap_width,save_bootstrap_freq)

            # estimated_uncertainty = self.jackknife_uncertainty_estimation(
            #     t_observed=t_observed, y_observed=y_observed, uncertainties=uncertainties, best_freq=final_best_freq,
            #     n_jobs=n_jobs,jackknife_points=bootstrap_points,jackknife_width=bootstrap_width,save_bootstrap_freq=save_bootstrap_freq
            # )

        return final_best_freq, estimated_uncertainty, combined_result_dict



    def check_frequency_significance(self, freq, objective_values, freq_range, **kwargs):
        """
        Check the significance of a frequency, its double (x2), and half (/2).

        Parameters:
        -----------
        freq : float
            The initial best frequency to check.
        objective_values : array_like
            Objective values for each frequency in the range.
        freq_range : array_like
            The range of frequencies to consider.
        search_width : float, optional
            The width for searching around the significant frequency.

        Returns:
        --------
        tuple
            The frequencies to further explore based on significance.
        """

        # Update class-level parameters if any
        self.params.update(kwargs)

        significance_threshold = np.percentile(objective_values, .1)

        # Check significance of the main frequency, x2, and /2
        frequencies_to_check = [freq, freq * 2, freq / 2]

        # Initialize an empty list to store significant frequencies with labels
        significant_freqs_with_labels = []

        for test_freq in frequencies_to_check:
            test_freq_idx = np.abs(freq_range - test_freq).argmin()
            test_freq_obj_val = objective_values[test_freq_idx]

            if test_freq_obj_val < significance_threshold:
                label = 'x2' if test_freq == freq * 2 else '/2' if test_freq == freq / 2 else 'best'
                significant_freqs_with_labels.append((label, test_freq))
                print(f"Frequency {test_freq} labeled as '{label}' is significant and will be further explored.")

        if not significant_freqs_with_labels:
            print('No significant multiples found. Considering the initial frequency with caution.')
            significant_freqs_with_labels.append(('best', freq))

        return significant_freqs_with_labels



    def bootstrap_uncertainty_estimation(self, t_observed, y_observed, uncertainties, best_freq, n_bootstrap,
                                         n_jobs, bootstrap_points, bootstrap_width, save_bootstrap_freq, **kwargs):
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
        bootstrap_points : int
            Number of points to use in the tight frequency check.
        bootstrap_width : float
            The width of the search range in the tight check.

        Returns:
        --------
        float
            The estimated true uncertainty.
        """
        self.params.update(kwargs)

        # Generate a tight frequency range around the best frequency
        tight_freq_range = np.linspace(best_freq * (1 - bootstrap_width), best_freq * (1 + bootstrap_width),
                                       bootstrap_points)



        def bootstrap_task(_):
            if bootstrap_points <= 0:
                raise ValueError(f"Invalid number of tight check points: {bootstrap_points}. Must be positive.")

            t_resampled, y_resampled, uncertainties_resampled = resample(t_observed, y_observed, uncertainties)

            def task_for_each_frequency(freq):
                result = self.nonparametric_kernel_regression(t_observed=t_resampled, y_observed=y_resampled,
                                                              uncertainties=uncertainties_resampled,
                                                              freq=freq,
                                                              kernel_type=self.params['kernel_type'],
                                                              regression_type=self.params['regression_type'],
                                                              bandwidth_method=self.params['bandwidth_method'],
                                                              alpha=self.params['alpha'],
                                                              use_grid=self.params['use_grid'])
                return result[4]  # Returning squared_residual and residuals

            # Evaluate the small grid around the best frequency
            with Parallel(n_jobs=n_jobs) as parallel:
                tight_results = parallel(delayed(task_for_each_frequency)(f) for f in tight_freq_range)

            # Find the frequency with the smallest residual in the resampled data
            min_freq = tight_freq_range[np.argmin(tight_results)]
            return min_freq


        # Bootstrap sampling
        bootstrap_freqs = [bootstrap_task(_) for _ in range(n_bootstrap)]

        if self.params['show_bootstrap_histogram']:
            plt.figure(figsize=(8, 8))

            # Create histogram and capture bin heights
            counts, bins, _ = plt.hist(bootstrap_freqs, bins=100)

            # Determine the height of the tallest bin
            max_height = max(counts)

            # Set x-axis limits based on the tight_freq_range
            min_freq = np.min(tight_freq_range)
            max_freq = np.max(tight_freq_range)
            plt.xlim(min_freq, max_freq)

            # Plot vertical line at best_freq
            plt.axvline(x=best_freq, color='r', linewidth=2, ymax=max_height / plt.ylim()[1])

            plt.title('KDE of Bootstrap Frequencies')
            plt.xlabel('Frequency')
            plt.ylabel('Density')
            plt.show()

        # Calculate the standard deviation of the best frequencies as the uncertainty measure
        estimated_uncertainty = np.std(bootstrap_freqs)

        if save_bootstrap_freq:
            return bootstrap_freqs
        else:
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
                    t_observed, y_observed, uncertainties, freq_list,  **kwargs
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

