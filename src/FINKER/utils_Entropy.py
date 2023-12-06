### LOCAL LINEAR REGRESSION AND LOCAL CONSTANT REGRESSION
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from sklearn.utils import resample


# Correcting the indentation and finishing the class definition

class EntropyFunctions:
    def __init__(self, M=10, N=10):
        """
        Initialize the EntropyFunctions class with default bin sizes for phase and magnitude.

        Parameters:
        - M (int): Number of bins for phase. Default is 10.
        - N (int): Number of bins for magnitude. Default is 10.
        """
        self.M = M
        self.N = N

    @staticmethod
    def conditional_entropy(phase, y, M, N):
        """
        Calculate the conditional entropy of the phase-folded light curve.

        Parameters:
        - phase (array-like): Phase-folded time values
        - y (array-like): The observed y-values of the light curve
        - M, N (int): Number of bins for phase and magnitude

        Returns:
        - float: The conditional entropy
        """
        phase_bins = np.linspace(0, 1, M + 1)
        mag_bins = np.linspace(min(y), max(y), N + 1)
        hist2d, _, _ = np.histogram2d(phase, y, bins=(phase_bins, mag_bins))
        p_xi_yj = hist2d / np.sum(hist2d)
        hist1d, _ = np.histogram(phase, bins=phase_bins)
        p_xi = hist1d / np.sum(hist1d)
        H = -np.sum(p_xi_yj * np.log((p_xi_yj + 1e-10) / (p_xi[:, None] + 1e-10)))
        return H

    def perform_conditional_entropy_phase_folding(self, t, y_observed, freq, show_plot=False):
        """
        Phase fold the light curve and calculate its conditional entropy.

        Parameters:
        - t (array-like): The time values of the light curve
        - y_observed (array-like): The observed y-values of the light curve
        - freq (float): The frequency for phase folding
        - show_plot (bool, optional): Whether to show a plot. Default is False.

        Returns:
        - tuple: Phase-folded time values and the conditional entropy
        """
        # Phase fold the light curve
        phase = (t * freq) % 1

        # Calculate conditional entropy
        H = self.conditional_entropy(phase, y_observed, self.M, self.N)

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

    def objective_function_entropy(self, freq, t, y_observed):
        """
        Objective function to be minimized using conditional entropy.

        Parameters:
        - freq (float): Frequency for phase folding
        - t (array-like): The time values of the light curve
        - y_observed (array-like): The observed y-values of the light curve

        Returns:
        - float: The conditional entropy
        """
        _, H = self.perform_conditional_entropy_phase_folding(t, y_observed, freq)
        return H  # Return conditional entropy

    def parallel_conditional_entropy(self, t_observed, y_observed, freq_list, n_jobs=-2, verbose=0,
                                     search_width=0.001, enable_tight_check=False, tight_check_points=1000,
                                     n_bootstrap=1000, bootstrap_points=100,bootstrap_width=0.01, show_bootstrap_histogram=False):
        """
        Calculate conditional entropy for a list of frequencies in parallel.

        Parameters:
        - t_observed (array-like): The observed time values of the light curve
        - y_observed (array-like): The observed y-values of the light curve
        - freq_list (array-like): List of frequencies for phase folding
        - n_jobs (int, optional): The number of jobs to run in parallel. Default is -2.
        - verbose (int, optional): Verbosity level for parallel processing. Default is 0.
        - search_width (float, optional): Width of the focused search around the best frequency. Default is 0.001.
        - enable_tight_check (bool, optional): Whether to perform a tighter grid search. Default is False.

        Returns:
        - tuple: Best frequency, entropy, and a dictionary of frequencies and their entropies
        """

        def task_for_each_frequency(freq):
            _, H = self.perform_conditional_entropy_phase_folding(t_observed, y_observed, freq)
            return H

        # Broad frequency search
        with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
            entropy_results = parallel(delayed(task_for_each_frequency)(freq) for freq in freq_list)

        broad_result_dict = dict(zip(freq_list, entropy_results))

        combined_result_dict = broad_result_dict.copy()

        if enable_tight_check:
            # Tighter grid search around the best frequency from broad search
            best_freq_broad = min(broad_result_dict, key=broad_result_dict.get)
            tight_freq_range = np.linspace(best_freq_broad * (1 - search_width), best_freq_broad * (1 + search_width),
                                           tight_check_points)
            with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
                tight_entropy_results = parallel(delayed(task_for_each_frequency)(freq) for freq in tight_freq_range)

            tight_result_dict = dict(zip(tight_freq_range, tight_entropy_results))

            # Combine results from both searches
            combined_result_dict.update(tight_result_dict)

        # Find the best frequency from combined results
        best_freq_combined = min(combined_result_dict, key=combined_result_dict.get)


        # Bootstrap uncertainty estimation with residuals at optimal frequency
        estimated_uncertainty = self.bootstrap_uncertainty_estimation_entropy(
            t_observed=t_observed, y_observed=y_observed, best_freq=best_freq_combined, n_bootstrap=n_bootstrap,
            n_jobs=n_jobs,bootstrap_width=bootstrap_width, bootstrap_points=bootstrap_points,show_bootstrap_histogram=show_bootstrap_histogram
        )

        return best_freq_combined, estimated_uncertainty, combined_result_dict




    def bootstrap_uncertainty_estimation_entropy(self, t_observed, y_observed, best_freq, n_bootstrap,
                                                 n_jobs=-2, bootstrap_width=0.05, bootstrap_points=1000,
                                                 verbose=0,
                                                 show_bootstrap_histogram=False):
        """
        Perform bootstrap sampling to estimate uncertainties in the best frequency based on conditional entropy.

        Parameters:
        - t_observed (array-like): The observed time values of the light curve.
        - y_observed (array-like): The observed y-values (e.g., magnitude) of the light curve.
        - best_freq (float): The best frequency determined from the previous analysis.
        - n_bootstrap (int): The number of bootstrap samples to generate.
        - n_jobs (int): The number of jobs to run in parallel.
        - bootstrap_width (float): The width of the search range in the tight check.
        - verbose (int): Verbosity level.

        Returns:
        - float: The estimated uncertainty in the best frequency.
        """
        # Generate a frequency range around the best frequency
        tight_freq_range = np.linspace(best_freq * (1 - bootstrap_width), best_freq * (1 + bootstrap_width), bootstrap_points)

        def bootstrap_task(_):
            # Resample the observed data
            t_resampled, y_resampled = resample(t_observed, y_observed)

            def task_for_each_frequency(freq):
                _, H = self.perform_conditional_entropy_phase_folding(t_resampled, y_resampled, freq)
                return H

            # Find the frequency with the smallest entropy in the resampled data
            with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
                entropy_results = parallel(delayed(task_for_each_frequency)(f) for f in tight_freq_range)

            return tight_freq_range[np.argmin(entropy_results)]

        # Bootstrap sampling
        bootstrap_freqs = [bootstrap_task(_) for _ in range(n_bootstrap)]

        if  show_bootstrap_histogram:
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

        return estimated_uncertainty




