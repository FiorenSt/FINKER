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
best_freq, freq_err, result_dict = finker.parallel_nonparametric_kernel_regression(
    t_observed=t_observed, 
    y_observed=y_observed,
    uncertainties=uncertainties,
    freq_list=freq,
    show_plot=False,
    kernel_type='gaussian',
    regression_type='local_constant',
    bandwidth_method='custom',
    n_jobs=-2,
    verbose=1,
    search_width=0.001, #once the best frequency (or multiple potential ones) has been found, it will better search in a 0.1% range from that frequency
    tight_check_points=1000,  #number of points used in the tight search
    estimate_uncertainties=True, #enabling the uncertainties estimate procedure
    n_bootstrap=1000, #number of repetitions of the procedure
    bootstrap_width=0.005 #grid size of .5% around the best frequency to evaluate the uncertainties
    bootstrap_points=100, #number of points used for the evaluation
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
