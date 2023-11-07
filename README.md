

![FINKER Logo](https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjo3NywidyI6MTAwMCwiZnMiOjc3LCJmZ2MiOiIjQjgxODIwIiwiYmdjIjoiI0ZGRkZGRiIsInQiOjF9/RklOS0VS/kg-second-chances-sketch.png)

# Frequency Identification with Kernel Regression

## Overview

FINKER (Frequency Identification with KErnel Regression) is an advanced statistical tool for optimal frequency identification using nonparametric kernel regression. It is designed to offer more accurate estimation in various contexts, including handling measurement uncertainties and facilitating multiband processing, even with as few as 10 observations. The project leverages the power of kernel regression methods to provide robust, flexible analysis, particularly valuable in astronomical and statistical applications.

## Features

- **Local Linear and Constant Regression**: Utilizes local regression techniques for precise modeling.
- **Kernel Functions**: Includes Gaussian, periodic, and locally periodic kernels for diverse applications.
- **Bandwidth Calculation**: Implements a custom fixed bandwidth for computational efficiency. Multiple other options are available: Silverman's rule, Scott's rule, and an Adaptive bandwidth.
- **Multiband Processing**: Capable of handling multi-frequency data efficiently.
- **Error Handling**: Robust to measurement uncertainties, enhancing reliability in real-world datasets.

## Installation

To use FINKER, clone the repository and install the required dependencies:

```bash
git clone https://github.com/FiorenSt/FINKER/
cd FINKER
pip install -r requirements.txt
```

## Usage

Here's a simple example to demonstrate the use of FINKER's kernel regression functions:

```python
from FINKER.src.utils import nonparametric_kernel_regression

# Example usage of the Gaussian kernel
result = nonparametric_kernel_regression(t_observed, y_observed, y_uncertainties, freq)
```

Replace `t_observed` and `y_observed` with your observations and `freq` with the folding frequency.

## Contributing

Contributions to FINKER are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

This project is licensed under the [Apache License](LICENSE) - see the LICENSE file for details.

## Contact

For any queries or further information, please reach out to [Your Contact Information].
