# Example Script for Demonstrating Usage of Functions in utils.py
import numpy as np
import matplotlib.pyplot as plt
from src.utils_SYNTHETIC_LIGHTCURVES import *
from src.utils_FINKER import *


# Generate a synthetic light curve
t_observed, y_observed, uncertainties = SyntheticLightCurveGenerator.generate_synthetic_light_curve(
    n_points=500, time=10, freq_primary=1, amplitude_primary=1, freq_secondary=1,
    amplitude_secondary=0, eclipse_depth=0, baseline_magnitude=17.0,
    noise_function=None, n_repeats=10, random_seed=42)


# Plotting the light curve
plt.figure(figsize=(10, 6))
plt.errorbar(t_observed, y_observed, yerr=uncertainties, fmt='o', markersize=5)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Synthetic Light Curve')
plt.gca().invert_yaxis()  # Inverting y-axis for magnitude
plt.show()


# Set up the frequency range for the search
freq = np.linspace(0.001,2,1000)

# Creating a FINKER instance
finker = FINKER()

# Running a parallel FINKER search
best_freq, freq_err, _, result_dict, _ = finker.parallel_nonparametric_kernel_regression(
    t_observed=t_observed,
    y_observed=y_observed,
    freq_list=freq,
    uncertainties=None,
    show_plot=False,
    kernel_type='gaussian',
    regression_type='local_constant',
    bandwidth_method='custom',
    use_grid = True,
    n_jobs=-2
)


# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(list(result_dict.keys()), list(result_dict.values()), label='Objective Value', s=1)
ax.axvline(x=best_freq, color='g', linestyle='--', label=f'Best Frequency: {best_freq:.7f}', lw=2)
ax.axvline(x=1, color='black', linestyle='--', label=f'True Frequency: {1:.7f}', lw=2)
ax.set_xlabel('Frequency', fontsize=18)
ax.set_ylabel('Squared Residuals', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(loc='upper left', fontsize=14)
