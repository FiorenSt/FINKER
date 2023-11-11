### LOCAL LINEAR REGRESSION AND LOCAL CONSTANT REGRESSION
import matplotlib.pyplot as plt
from joblib import Parallel, delayed





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

    def parallel_conditional_entropy(self, t_observed, y_observed, freq_list, n_jobs=-2, verbose=0, search_width=0.01):
        """
        Calculate conditional entropy for a list of frequencies in parallel.

        Parameters:
        - t_observed (array-like): The observed time values of the light curve
        - y_observed (array-like): The observed y-values of the light curve
        - freq_list (array-like): List of frequencies for phase folding
        - n_jobs (int, optional): The number of jobs to run in parallel. Default is -2.
        - verbose (int, optional): Verbosity level for parallel processing. Default is 0.
        - search_width (float, optional): Width of the focused search around the best frequency. Default is 0.01.

        Returns:
        - tuple: Best frequency, entropy, and a dictionary of frequencies and their entropies
        """

        def task_for_each_frequency(freq):
            # Use the instance attributes M and N directly inside the method
            _, H = self.perform_conditional_entropy_phase_folding(t_observed, y_observed, freq)
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





import numpy  as np
from UNUSED_CODE.utils import generate_synthetic_light_curve

# Generate synthetic light curve
t_observed, y_observed, uncertainties = generate_synthetic_light_curve(n_points=75,
                                                                       freq_primary=1, amplitude_primary=.1, time=10,
                                                                       freq_secondary=0, amplitude_secondary=0,
                                                                       eclipse_depth=0,
                                                                                           random_seed=5)


freq_list = np.linspace(0,2,100)

entropy=EntropyFunctions()

best_freq, best_entropy, results = entropy.parallel_conditional_entropy(t_observed, y_observed,freq_list)