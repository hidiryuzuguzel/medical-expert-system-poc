import unittest
import numpy as np
from inference_engine import make_engine


class TestInferenceEngine(unittest.TestCase):
    def test_make_engine_for_dimension_mismatch(self):
        D = np.random.rand(20, 10) < 0.3 # Diseases/Symptom matrix

        with self.assertRaises(AssertionError):
                eng = make_engine(D,
                                  theta0=0.99,
                                  theta=0.02 * np.ones((20, 1)),
                                  pd = 0.01 * np.ones((15,1)))

    def test_make_engine_for_nonbinary_influence_matrix(self):
        D = np.random.rand(20, 10)
        with self.assertRaises(AssertionError):
                eng = make_engine(D,
                                  theta0=0.99,
                                  theta=0.02 * np.ones((20, 1)),
                                  pd = 0.01 * np.ones((10,1)))

    def test_if_output_of_make_engine_is_dict(self):
        D = np.random.rand(20, 10) < 0.3  # Diseases/Symptom matrix
        eng = make_engine(D,
                          theta0=0.99,
                          theta=0.02 * np.ones((20, 1)),
                          pd=0.01 * np.ones((10, 1)))
        self.assertIsInstance(eng, dict)
        self.assertListEqual(list(eng.keys()), ['S', 'N', 'pd', 'D', 'pSD', 'th0', 'th', 'dn', 'sn'])
        self.assertEqual(len(eng['dn']), 10)
        self.assertListEqual(eng['dn'], ['Disease%d'%(i+1) for i in range(10)])
        self.assertEqual(len(eng['sn']), 20)
        self.assertListEqual(eng['sn'], ['Symptom%d'%(j+1) for j in range(20)])





if __name__ == '__main__':
    unittest.main()
