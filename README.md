

![FINKER Logo](https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjo3NywidyI6MTAwMCwiZnMiOjc3LCJmZ2MiOiIjQjgxODIwIiwiYmdjIjoiI0ZGRkZGRiIsInQiOjF9/RklOS0VS/kg-second-chances-sketch.png)


## Overview

FINKER (Frequency Identification through Nonparametric Kernel Regression) is an advanced statistical tool for optimal frequency identification using nonparametric kernel regression. It is designed to offer more accurate estimation in various contexts, including handling measurement uncertainties and facilitating multiband processing, even with as few as 10 observations. The project leverages the power of kernel regression methods to provide robust, flexible analysis, particularly valuable in astronomical and statistical applications.

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
from FINKER.src.utils_FINKER import *

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
    n_jobs=-2
)
```

Replace `t_observed`, `y_observed`, and `uncertainties` with your observations in numpy arrays. 
Substitute `freq` with the range of folding frequencies you want to test. A fine grid with 0.0001 distance between grid points is enough for most cases.

## Contributing

Contributions to FINKER are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

This project is licensed under the [Apache License](LICENSE) - see the LICENSE file for details.

## Contact

For any queries or further information, please reach out to f.stoppa@astro.ru.nl.
