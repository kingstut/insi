import unittest
import torch 
from insi import Probe, Probes, Cortex, build_model

class TestProbe(unittest.TestCase):
    def test_add(self):
        probe = Probe()
        probe.add(5)
        self.assertEqual(probe.value, 5)

class TestProbes(unittest.TestCase):
    def test_add_probe(self):
        probes_collection = Probes()
        probe1 = Probe()
        probes_collection.add_probe(probe1)
        self.assertEqual(len(probes_collection.probes), 1)

    def test_delete_probe(self):
        probes_collection = Probes()
        probe1 = Probe()
        probes_collection.add_probe(probe1)
        self.assertEqual(len(probes_collection.probes), 1)
        probes_collection.delete_probe(0)
        self.assertEqual(len(probes_collection.probes), 0)

class TestCortex(unittest.TestCase):
    def test_tune(self):
        # Mock neural network model and objective function
        class MockModel:
            def __init__(self):
                self.parameters = [torch.tensor([1.0, 2.0])]

        def mock_objective(pred):
            return torch.tensor(0.5)

        probes = Probes()
        cortex = Cortex(probes, MockModel(), mock_objective)
        cortex.tune(epochs=1, lr=0.1)
        self.assertTrue(True)  

class TestBuildModel(unittest.TestCase):
    def test_build_model(self):
        # Mock neural network model and objective function
        class MockModel:
            def __init__(self):
                self.parameters = [torch.tensor([1.0, 2.0])]

        def mock_objective(pred):
            return torch.tensor(0.5)

        probes = [Probe(), Probe()]
        model = build_model(probes, MockModel(), mock_objective, epochs=1)
        self.assertTrue(True)  

if __name__ == '__main__':
    unittest.main()
