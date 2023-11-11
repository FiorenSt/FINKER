
import unittest
import numpy as np
from src.utils import gaussian_kernel, periodic_kernel, locally_periodic_kernel, silverman_bw, custom_bw, heuristic_median_bw, adaptive_bw

class TestKernelFunctions(unittest.TestCase):

    def test_gaussian_kernel(self):
        u = 0
        l_squared = 1
        expected_result = 1
        self.assertAlmostEqual(gaussian_kernel(u, l_squared), expected_result, places=5)

    def test_silverman_bw(self):
        data = np.array([1, 2, 3, 4, 5])
        # Expected result calculated based on Silverman's rule
        expected_result = 1.06 * data.std() * len(data) ** (-1/5)
        self.assertAlmostEqual(silverman_bw(data), expected_result, places=5)

    def test_custom_bw(self):
        data = np.array([1, 2, 3, 4, 5])
        alpha = 0.9
        # Expected result calculated based on custom rule
        expected_result = alpha * len(data) ** (-1/5)
        self.assertAlmostEqual(custom_bw(data, alpha), expected_result, places=5)

    def test_adaptive_bw(self):
        data = np.array([[1], [2], [3], [4], [5]])
        k = 2
        # Test if function returns the correct length of bandwidths and an instance of NearestNeighbors
        avg_bandwidths, nbrs = adaptive_bw(data, k)
        self.assertEqual(len(avg_bandwidths), len(data))
        self.assertIsInstance(nbrs, NearestNeighbors)

# Additional tests for other functions can be added here

if __name__ == '__main__':
    unittest.main()
