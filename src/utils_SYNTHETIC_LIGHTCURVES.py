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
    def get_mag_stddev_relation(upper=13, lower=20, degree=5):

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

        fx = np.polynomial.Polynomial.fit(xx, yq, degree, domain=[0, 1], window=[0, 1])

        ## !! returns function !!
        return fx

    @staticmethod
    def generate_synthetic_light_curve(n_points=500, time=10, freq_primary=1,
                                       amplitude_primary=1, freq_secondary=1,
                                       amplitude_secondary=0, eclipse_depth=0,
                                       baseline_magnitude=17.0, noise_function=None,
                                       n_repeats=10, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        if noise_function is None:
            noise_function = SyntheticLightCurveGenerator.get_mag_stddev_relation()
        noise_std = noise_function(baseline_magnitude)

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
        # Simulate repeated observations
        repeated_observations = np.zeros((n_repeats, n_points))
        for i in range(n_repeats):
            uncertainties = np.random.normal(0, noise_std, n_points)
            repeated_observations[i, :] = y_magnitude + uncertainties

        # Calculate the mean and standard deviation of the repeated observations
        y_magnitude_observed = np.mean(repeated_observations, axis=0)
        sigma = np.std(repeated_observations, axis=0)
        return t_observed, y_magnitude_observed, sigma






