
# Example Script for Demonstrating Usage of Functions in utils.py

import numpy as np
import matplotlib.pyplot as plt
from src.utils import gaussian_kernel, periodic_kernel, locally_periodic_kernel, silverman_bw

# Example data for demonstration
x_data = np.linspace(-5, 5, 100)

# Gaussian Kernel Demonstration
l_squared = 2.0
gaussian_values = gaussian_kernel(x_data, l_squared)

# Periodic Kernel Demonstration
l = 1.0
p = 2.0
periodic_values = periodic_kernel(x_data, l, p)

# Locally Periodic Kernel Demonstration
locally_periodic_values = locally_periodic_kernel(x_data, l, p)

# Plotting the Kernel Functions
plt.figure(figsize=(12, 8))
plt.plot(x_data, gaussian_values, label='Gaussian Kernel')
plt.plot(x_data, periodic_values, label='Periodic Kernel')
plt.plot(x_data, locally_periodic_values, label='Locally Periodic Kernel')
plt.xlabel('x')
plt.ylabel('Kernel Value')
plt.title('Kernel Functions Demonstration')
plt.legend()
plt.show()

# Bandwidth Calculation Demonstration
data = np.random.normal(0, 1, 100)
bandwidth = silverman_bw(data)
print(f"Calculated bandwidth using Silverman's rule: {bandwidth}")

# Additional examples demonstrating other functions can be added here
