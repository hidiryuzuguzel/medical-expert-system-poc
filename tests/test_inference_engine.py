import unittest
import numpy as np
from inference_engine import make_engine, infer_best_k


class TestInferenceEngine(unittest.TestCase):

    def setUp(self):
        self.D = np.array([[1, 0], [1, 1], [0, 1]]) # 2 diseases & 3 symptoms Toy Example
        self.pd = 0.01 * np.ones((self.D.shape[1], 1))
        self.theta0 = 0.99
        self.theta = 0.02 * np.ones((self.D.shape[0], 1))
        self.eng = make_engine(self.D, theta0=self.theta0, theta=self.theta, pd=self.pd)
        self.no_disease_conf = [0, 0]
        self.only_disease1_conf = [1, 0]
        self.only_disease2_conf = [0, 1]
        self.disease1_and_disease2_conf = [1, 1]

    def test_make_engine_for_dimension_mismatch(self):

        with self.assertRaises(AssertionError):
                eng = make_engine(self.D,
                                  theta0=self.theta0,
                                  theta=0.02 * np.ones((20, 1)),
                                  pd = 0.01 * np.ones((15,1)))

    def test_make_engine_for_nonbinary_influence_matrix(self):
        D = np.random.rand(3, 2)
        with self.assertRaises(AssertionError):
                eng = make_engine(D,
                                  theta0=self.theta0,
                                  theta=self.theta,
                                  pd = self.pd)

    def test_if_output_of_make_engine_is_dict(self):

        self.assertIsInstance(self.eng, dict)
        self.assertListEqual(list(self.eng.keys()), ['S', 'N', 'pd', 'D', 'pSD', 'th0', 'th', 'dn', 'sn'])
        self.assertListEqual(self.eng['D'].tolist(), self.D.tolist())
        self.assertEqual(self.eng['th0'], self.theta0)
        self.assertListEqual(self.eng['th'].tolist(), self.theta.tolist())
        self.assertListEqual(self.eng['pd'].tolist(), self.pd.tolist())
        self.assertEqual(len(self.eng['dn']), 2)
        self.assertListEqual(self.eng['dn'], ['Disease%d'%(i+1) for i in range(2)])
        self.assertEqual(len(self.eng['sn']), 3)
        self.assertListEqual(self.eng['sn'], ['Symptom%d'%(j+1) for j in range(3)])

    def test_inference_no_observation(self):

        conf, logPconf = infer_best_k(self.eng, sidx=[], so=[], best_k=4, MX=2)
        actual_posterior = self.calculate_posterior(conf, logPconf)
        expected_posterior = [0.9801, 0.0099, 0.0099, 0.0001]

        self.assertListEqual(expected_posterior, actual_posterior)

    def test_inference_symptom1_present(self):

        conf, logPconf = infer_best_k(self.eng, sidx=[0], so=[1], best_k=4, MX=2)
        actual_posterior = self.calculate_posterior(conf, logPconf)
        expected_posterior = [0.4975, 0.4925, 0.0050, 0.0050]

        self.assertListEqual(expected_posterior, actual_posterior)

    def test_inference_symptom1_present_and_symptom2_present(self):

        conf, logPconf = infer_best_k(self.eng, sidx=[0, 1], so=[1, 1], best_k=4, MX=2)
        actual_posterior = self.calculate_posterior(conf, logPconf)
        expected_posterior = [0.0100, 0.9701, 0.0099, 0.0100]

        self.assertListEqual(expected_posterior, actual_posterior)

    def test_inference_symptom1_present_and_symptom2_present_and_symptom3_present(self):

        conf, logPconf = infer_best_k(self.eng, sidx=[0, 1, 2], so=[1, 1, 1], best_k=4, MX=2)
        actual_posterior = self.calculate_posterior(conf, logPconf)
        expected_posterior = [0.0034, 0.3311, 0.3311, 0.3343]

        self.assertListEqual(expected_posterior, actual_posterior)

    def test_inference_symptom1_present_and_symptom2_present_and_symptom3_absent(self):

        conf, logPconf = infer_best_k(self.eng, sidx=[0, 1, 2], so=[1, 1, 0], best_k=4, MX=2)
        actual_posterior = self.calculate_posterior(conf, logPconf)
        expected_posterior = [0.0102, 0.9894, 0.0002, 0.0002]

        self.assertListEqual(expected_posterior, actual_posterior)

    def calculate_posterior(self, conf, logPconf):

        idx_no_disease_conf = np.where((conf == self.no_disease_conf).all(axis=1))[0][0]
        idx_only_disease1_conf = np.where((conf == self.only_disease1_conf).all(axis=1))[0][0]
        idx_only_disease2_conf = np.where((conf == self.only_disease2_conf).all(axis=1))[0][0]
        idx_disease1_and_disease2_conf = np.where((conf == self.disease1_and_disease2_conf).all(axis=1))[0][0]

        posterior = np.exp(logPconf) / np.sum(np.exp(logPconf))
        actual_posterior = posterior[[idx_no_disease_conf, idx_only_disease1_conf,
                                      idx_only_disease2_conf, idx_disease1_and_disease2_conf]].round(4).tolist()

        return actual_posterior


if __name__ == '__main__':
    unittest.main()
