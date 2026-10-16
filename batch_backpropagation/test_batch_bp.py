import unittest
import numpy as np
from nn_sandbox.backend.algorithms.batch_bp import BatchBP, Config, xor_data, parse_data


class BatchBPTests(unittest.TestCase):
    def test_gradient_matches_finite_differences_multiple_outputs(self):
        rng = np.random.default_rng(5)
        for activation in ('unipolar', 'bipolar'):
            model = BatchBP(Config(inputs=3, hidden=4, outputs=2, activation=activation))
            x = rng.normal(size=(5, 3))
            y = rng.uniform(0, 1, (5, 2))
            if activation == 'bipolar':
                y = 2 * y - 1
            gradients = model.gradients(x, y)
            for weights, analytic in zip((model.V, model.W), gradients):
                numerical = np.zeros_like(weights)
                for index in np.ndindex(weights.shape):
                    original = weights[index]
                    weights[index] = original + 1e-6
                    plus = model.metrics(x, y, 0)['error']
                    weights[index] = original - 1e-6
                    minus = model.metrics(x, y, 0)['error']
                    weights[index] = original
                    numerical[index] = (plus - minus) / 2e-6
                np.testing.assert_allclose(analytic, numerical, atol=2e-9, rtol=2e-6)

    def test_one_epoch_is_full_batch_update(self):
        cfg = Config(max_epochs=1, threshold=1e-10)
        model = BatchBP(cfg)
        x, y = xor_data()
        v, w = model.V.copy(), model.W.copy()
        dv, dw = model.gradients(x, y)
        model.fit(x, y)
        np.testing.assert_allclose(model.V, v - cfg.learning_rate * dv)
        np.testing.assert_allclose(model.W, w - cfg.learning_rate * dw)
        self.assertIn('未满足', model.status)

    def test_both_activations_learn_xor(self):
        for activation in ('unipolar', 'bipolar'):
            x, y = xor_data(activation)
            model = BatchBP(Config(activation=activation))
            model.fit(x, y)
            self.assertLessEqual(model.history[-1]['rmse'], 0.05)
            cutoff = 0.5 if activation == 'unipolar' else 0
            np.testing.assert_array_equal(model.predict(x) > cutoff, y > cutoff)

    def test_seed_initialization_and_bias(self):
        a, b = BatchBP(Config()), BatchBP(Config())
        for first, second in ((a.V, b.V), (a.W, b.W)):
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.all((first >= -1) & (first <= 1)))
        self.assertEqual(a.V.shape, (3, 4))
        self.assertEqual(a.W.shape, (5, 1))

    def test_data_errors_and_multiple_outputs(self):
        x, y = parse_data('0,1,0,1\n1 0 1 0', 2, 2)
        BatchBP(Config(outputs=2, max_epochs=2)).fit(x, y)
        with self.assertRaises(ValueError):
            parse_data('0,1', 2, 1)
        with self.assertRaises(ValueError):
            BatchBP(Config()).fit([[0, 1]], [[-1]])
        with self.assertRaises(ValueError):
            BatchBP(Config()).fit([[float('nan'), 0]], [[0]])
        with self.assertRaises(ValueError):
            BatchBP(Config(learning_rate=0))

    def test_cancel_and_initial_threshold(self):
        x, y = xor_data()
        model = BatchBP(Config())
        model.fit(x, y, should_stop=lambda: True)
        self.assertEqual(model.status, '用户停止')
        self.assertEqual(model.history[-1]['epoch'], 0)
        model = BatchBP(Config(threshold=2))
        model.fit(x, y)
        self.assertEqual(model.status, '达到误差阈值')
        self.assertEqual(model.history[-1]['epoch'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
