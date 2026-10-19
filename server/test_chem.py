import unittest

from .chem import ChemistryError, candidate_payload, molecule_payload, parse_smiles


class ChemistryTests(unittest.TestCase):
    def test_ethanol_descriptors_and_real_3d_geometry(self):
        molecule = molecule_payload("CCO")
        self.assertEqual(molecule["descriptors"]["formula"], "C2H6O")
        self.assertTrue(molecule["geometry"]["has3d"])
        self.assertGreater(molecule["geometry"]["coordinateSpan"]["x"], 0.1)

    def test_invalid_smiles_fails_with_friendly_error(self):
        with self.assertRaises(ChemistryError):
            parse_smiles("not-a-molecule")

    def test_demo_candidate_is_canonical_and_valid(self):
        candidate = candidate_payload("OCC", "Ethanol", 0, "demo")
        self.assertTrue(candidate["valid"])
        self.assertEqual(candidate["smiles"], "CCO")


if __name__ == "__main__":
    unittest.main()
