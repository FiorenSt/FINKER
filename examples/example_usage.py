
# Example Script for Demonstrating Usage of Functions in utils.py

import numpy as np
import matplotlib.pyplot as plt
from src.utils_SYNTHETIC_LIGHTCURVES import *
from src.utils_FINKER import *
from src.utils_Entropy import *


# Generate a synthetic light curve
t_observed, y_magnitude_observed, sigma = SyntheticLightCurveGenerator.generate_synthetic_light_curve(
    n_points=50, time=10, freq_primary=1, amplitude_primary=1, freq_secondary=1,
    amplitude_secondary=0, eclipse_depth=0, baseline_magnitude=17.0,
    noise_function=None, n_repeats=10, random_seed=42)


# Plotting the light curve
plt.figure(figsize=(10, 6))
plt.errorbar(t_observed, y_magnitude_observed, yerr=sigma, fmt='o', markersize=5)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Synthetic Light Curve')
plt.gca().invert_yaxis()  # Inverting y-axis for magnitude
plt.show()


# Set up the frequency range for the search
freq = np.array(range(0.01,2,100))

# Creating a FINKER instance
finker = FINKER()

# Running a parallel FINKER search
best_freq, freq_err, result_dict = finker.parallel_nonparametric_kernel_regression(
    t_observed=t_observed,
    y_observed=y_observed,
    freq_list=freq,
    uncertainties=None,
    show_plot=False,
    kernel_type='gaussian',
    regression_type='local_constant',
    bandwidth_method='custom',
    n_jobs=-2
)

