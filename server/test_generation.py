import unittest

from .main import prepare_candidates


class CandidateFallbackTests(unittest.TestCase):
    def test_unrenderable_vae_output_gets_closest_related_reference(self):
        batch = prepare_candidates(["(("], "vae")
        candidate = batch["candidates"][0]
        self.assertTrue(candidate["valid"])
        self.assertTrue(candidate["relatedReference"])
        self.assertEqual(candidate["source"], "related-reference")
        self.assertEqual(candidate["originalOutput"], "((")
        self.assertEqual(batch["relatedCount"], 1)
        self.assertEqual(batch["modelMetrics"]["valid"], 0)
        self.assertEqual(batch["metrics"]["valid"], 1)

    def test_valid_vae_output_is_preserved_without_fallback(self):
        batch = prepare_candidates(["CCO"], "vae")
        candidate = batch["candidates"][0]
        self.assertTrue(candidate["valid"])
        self.assertFalse(candidate["relatedReference"])
        self.assertEqual(candidate["smiles"], "CCO")
        self.assertEqual(batch["modelMetrics"]["valid"], 1)


if __name__ == "__main__":
    unittest.main()
