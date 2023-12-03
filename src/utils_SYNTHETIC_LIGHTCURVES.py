import numpy as np
import pandas as pd
from astropy.table import Table
import warnings
from astropy.units import UnitsWarning
from numpy.polynomial import Polynomial

# Suppress UnitsWarning
warnings.filterwarnings('ignore', category=UnitsWarning)

class SyntheticLightCurveGenerator:
    def __init__(self):
        # Define the precomputed polynomial coefficients, domain, and window
        self.precomputed_coefficients = [-0.00311296, 0.11068129, -0.69244663, 2.01098336, -2.56210283, 1.29726758]
        self.domain = [13.79964161, 21.9997921]
        self.window = [0, 1]

    @staticmethod
    def sort_on_x(x, y):
        zipp = list(zip(x, y))
        zipp.sort(key=lambda x: x[0])
        x, y = list(zip(*zipp))
        return np.array(x), np.array(y)

    def get_mag_stddev_relation(self):
        # Returns a Polynomial object using the predefined coefficients
        return Polynomial(self.precomputed_coefficients, domain=self.domain, window=self.window)

    @staticmethod
    def magnitude_to_flux(magnitude, zero_point_flux=1.0):
        return zero_point_flux * np.power(10, -0.4 * (magnitude - 23.9))

    @staticmethod
    def calculate_flux_uncertainty(magnitude, magnitude_uncertainty, zero_point_flux=1.0):
        flux = SyntheticLightCurveGenerator.magnitude_to_flux(magnitude, zero_point_flux)
        return flux * np.log(10) * 0.4 * magnitude_uncertainty


    def generate_synthetic_light_curve(self, n_points=500, time=10, freq_primary=1,
                                       amplitude_primary=1, freq_secondary=1,
                                       amplitude_secondary=0, eclipse_depth=0,
                                       baseline_magnitude=17.0, random_seed=None,
                                       output_in_flux=True, zero_point_flux=1.0):
        if random_seed is not None:
            np.random.seed(random_seed)

        noise_function = self.get_mag_stddev_relation()

        # Time vector
        t_observed = np.sort(np.random.uniform(0, time, n_points))

        # Generate light curve components
        y_primary = - amplitude_primary * np.sin(2 * np.pi * freq_primary * t_observed)
        y_secondary = - amplitude_secondary * np.sin(2 * np.pi * freq_secondary * t_observed)
        y_eclipse = eclipse_depth * np.sin(2 * np.pi * freq_secondary * t_observed) ** 30
        y_magnitude = baseline_magnitude + y_primary + y_secondary + y_eclipse

        # Apply noise
        noise_std = noise_function(y_magnitude)
        uncertainties = np.random.normal(0, noise_std, n_points)
        y_magnitude_observed = y_magnitude + uncertainties

        # Output handling
        if output_in_flux:
            y_flux_observed = SyntheticLightCurveGenerator.magnitude_to_flux(y_magnitude_observed, zero_point_flux)
            flux_uncertainties = SyntheticLightCurveGenerator.calculate_flux_uncertainty(y_magnitude_observed, noise_std, zero_point_flux)
            return t_observed, y_flux_observed, np.abs(flux_uncertainties)
        else:
            return t_observed, y_magnitude_observed, noise_std
