import importlib.util
import unittest
from pathlib import Path

from .vae import ModelRuntime


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional model dependency")
class TrainedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts = Path(__file__).resolve().parent.parent / "trained_vae_artifacts"
        cls.runtime = ModelRuntime(artifacts / "vae_model.pt", artifacts / "vocab.json")

    def test_trained_artifact_loads_and_decodes(self):
        self.assertTrue(self.runtime.loaded, self.runtime.reason)
        self.assertEqual(self.runtime.latent_dim, 128)
        generated = self.runtime.generate(3, seed=42, temperature=0.65)
        self.assertEqual(len(generated), 3)

    def test_model_interpolation_decodes_all_steps(self):
        decoded, coordinates = self.runtime.interpolation("CCO", "CC(=O)O", steps=5)
        self.assertEqual(len(decoded), 5)
        self.assertEqual(coordinates.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
