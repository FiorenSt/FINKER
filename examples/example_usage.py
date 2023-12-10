from FINKER.utils_FINKER import *
from FINKER.utils_SYNTHETIC_LIGHTCURVES import *


# Create an instance of the class
generator = SyntheticLightCurveGenerator()

# Now call the method on this instance
t_observed, y_observed, uncertainties = generator.generate_synthetic_light_curve(
    n_points=100,
    time=10,
    freq_primary=0.6,
    snr_primary=7,
    baseline_magnitude=17.0,
    random_seed=5,
    output_in_flux=False,
    zero_point_flux=1.0
)


# Plotting the light curve
plt.figure(figsize=(10, 6))
plt.errorbar(t_observed, y_observed, yerr=uncertainties, fmt='o', markersize=5)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Synthetic Light Curve')
plt.gca().invert_yaxis()  # Inverting y-axis for magnitude
plt.show()



# Set up the frequency range for the search
freq_list = np.linspace(0.001,5,50000)

# Creating a FINKER instance
finker = FINKER()

# Running a parallel FINKER search
best_freq, freq_err, result_dict = finker.parallel_nonparametric_kernel_regression(
    t_observed=t_observed,
    y_observed=y_observed,
    uncertainties=uncertainties,
    freq_list=freq_list,
    show_plot=False,
    kernel_type='gaussian',
    regression_type='local_constant',
    bandwidth_method='custom',
    n_jobs=-2,
    tight_check_points=5000,
    search_width = 0.01,
    estimate_uncertainties=False,
    n_bootstrap= 500,
    bootstrap_width = 0.01,
    bootstrap_points= 500  
)


# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(list(result_dict.keys()), list(result_dict.values()), s=1)
ax.axvline(x=best_freq, color='g', linestyle='--', label=f'Best Frequency: {best_freq:.7f}', lw=2)
ax.axvline(x=1, color='black', linestyle='--', label=f'True Frequency: {1:.7f}', lw=2)
ax.set_xlabel('Frequency', fontsize=18)
ax.set_ylabel('Squared Residuals', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(loc='lower right', fontsize=14)


# Plot the light curve folded at the estimated frequency
phase, y_smoothed, _, _, bw, squared_residual = finker.nonparametric_kernel_regression(t_observed=t_observed,
                                                                                 y_observed=y_observed,
                                                                                 uncertainties=uncertainties,
                                                                                 freq= best_freq , kernel_type='gaussian',
                                                                                 regression_type='local_constant',
                                                                                 bandwidth_method='custom',
                                                                                 alpha=0.0618,
                                                                                 show_plot=True,
                                                                                 use_grid=True, grid_size=300)