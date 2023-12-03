import numpy as np
import pandas as pd
from astropy.table import Table
import warnings
from astropy.units import UnitsWarning

# Suppress UnitsWarning
warnings.filterwarnings('ignore', category=UnitsWarning)



class SyntheticLightCurveGenerator:
    def __init__(self):
        pass

    @staticmethod
    def sort_on_x(x, y):
        zipp = list(zip(x, y))
        zipp.sort(key=lambda x: x[0])
        x, y = list(zip(*zipp))

        return np.array(x), np.array(y)


    @staticmethod
    def get_mag_stddev_relation(upper=12, lower=22, degree=5):

        # Get magnitude standard deviation relation method
        dat = Table.read('DATA/16010_ML1_16010_q_20210428_red_cat.fits', format='fits')
        df = dat.to_pandas()
        df = df.loc[((df.MAG_OPT > upper) & (df.MAG_OPT < lower))]
        x = np.array(df.MAG_OPT.tolist())
        y = np.sqrt(1.) * np.array(df.MAGERR_OPT.tolist())
        xx, yy = SyntheticLightCurveGenerator.sort_on_x(x, y)
        dy = pd.Series(yy)
        yq = np.array(dy.rolling(500).quantile(0.25).tolist())

        idx = np.isfinite(yq)
        yq = yq[idx]
        xx = xx[idx]

        fx = np.polynomial.Polynomial.fit(xx, yq, degree, window=[0, 1])

        ## !! returns function !!
        return fx

    @staticmethod
    def magnitude_to_flux(magnitude, zero_point_flux=1.0):
        return zero_point_flux * np.power(10, -0.4 * (magnitude-23.9))

    @staticmethod
    def calculate_flux_uncertainty(magnitude, magnitude_uncertainty, zero_point_flux=1.0):
        flux = SyntheticLightCurveGenerator.magnitude_to_flux(magnitude, zero_point_flux)
        return flux * np.log(10) * 0.4 * magnitude_uncertainty

    @staticmethod
    def generate_synthetic_light_curve(n_points=500, time=10, freq_primary=1,
                                       amplitude_primary=1, freq_secondary=1,
                                       amplitude_secondary=0, eclipse_depth=0,
                                       baseline_magnitude=17.0,
                                       random_seed=None, output_in_flux=True, zero_point_flux=1.0):
        if random_seed is not None:
            np.random.seed(random_seed)

        noise_function = SyntheticLightCurveGenerator.get_mag_stddev_relation()

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


        noise_std = noise_function(y_magnitude)

        uncertainties = np.random.normal(0, noise_std, n_points)
        y_magnitude_observed = y_magnitude + uncertainties

        # Calculate the mean and standard deviation of the repeated observations
        sigma = noise_std

        if output_in_flux:
            # Convert magnitudes to flux
            y_flux_observed = SyntheticLightCurveGenerator.magnitude_to_flux(y_magnitude_observed, zero_point_flux)

            # Calculate flux uncertainties
            flux_uncertainties = SyntheticLightCurveGenerator.calculate_flux_uncertainty(y_magnitude_observed, sigma,
                                                                                         zero_point_flux)

            return t_observed, y_flux_observed, np.abs(flux_uncertainties)
        else:
            # Return in magnitude
            return t_observed, y_magnitude_observed, sigma





