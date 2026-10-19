"""Curated demo data for Molecule Studio.

These molecules are hand-curated examples, *not* samples from a trained VAE.
Every structure is still parsed and embedded in 3D with RDKit at request time.
"""

from __future__ import annotations

from typing import Dict, List

# A deliberately varied library of small, real molecules. The demo generator
# samples from this list so the interface is useful without a trained checkpoint.
DEMO_MOLECULES: List[Dict[str, str]] = [
    {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
    {"name": "Paracetamol", "smiles": "CC(=O)NC1=CC=C(O)C=C1"},
    {"name": "Vanillin", "smiles": "COc1cc(C=O)ccc1O"},
    {"name": "Glucose", "smiles": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"},
    {"name": "Serotonin", "smiles": "NCCc1c[nH]c2ccc(O)cc12"},
    {"name": "Dopamine", "smiles": "NCCc1ccc(O)c(O)c1"},
    {"name": "Naproxen", "smiles": "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1"},
    {"name": "Ketoprofen", "smiles": "CC(C(=O)O)c1cccc(c1)C(=O)c2ccccc2"},
    {"name": "Nicotine", "smiles": "CN1CCCC1c1cccnc1"},
    {"name": "Menthol", "smiles": "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"},
    {"name": "Limonene", "smiles": "CC1=CCC(CC1)C(=C)C"},
    {"name": "Eugenol", "smiles": "C=CCc1ccc(O)c(OC)c1"},
    {"name": "Indole", "smiles": "c1ccc2[nH]ccc2c1"},
    {"name": "Quinoline", "smiles": "c1ccc2ncccc2c1"},
    {"name": "Benzene", "smiles": "c1ccccc1"},
    {"name": "Phenol", "smiles": "Oc1ccccc1"},
    {"name": "Aniline", "smiles": "Nc1ccccc1"},
    {"name": "Toluene", "smiles": "Cc1ccccc1"},
    {"name": "Pyridine", "smiles": "c1ccncc1"},
    {"name": "Imidazole", "smiles": "c1cnc[nH]1"},
    {"name": "Furan", "smiles": "c1ccoc1"},
    {"name": "Thiophene", "smiles": "c1ccsc1"},
    {"name": "Salicylic acid", "smiles": "O=C(O)c1ccccc1O"},
    {"name": "Tyrosine", "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"},
    {"name": "Tryptophan", "smiles": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"},
    {"name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N"},
    {"name": "Sulfanilamide", "smiles": "Nc1ccc(cc1)S(N)(=O)=O"},
    {"name": "Acetone", "smiles": "CC(C)=O"},
    {"name": "Ethanol", "smiles": "CCO"},
    {"name": "Urea", "smiles": "NC(N)=O"},
]

# Chemically valid, vocabulary-compatible references used only when a raw VAE
# decode cannot be presented directly. Candidates are selected by SMILES edit
# distance to the raw output, so the structure stays as close as possible to
# the decoded sequence. These are never reported as model-generated molecules.
RELATED_MOLECULES: List[Dict[str, str]] = [
    {"name": "Methane", "smiles": "C"},
    {"name": "Ethane", "smiles": "CC"},
    {"name": "Ethylene", "smiles": "C=C"},
    {"name": "Propane", "smiles": "CCC"},
    {"name": "Propene", "smiles": "CC=C"},
    {"name": "Formaldehyde", "smiles": "C=O"},
    {"name": "Methanol", "smiles": "CO"},
    {"name": "Dimethyl ether", "smiles": "COC"},
    {"name": "Ethanol", "smiles": "CCO"},
    {"name": "Acetaldehyde", "smiles": "CC=O"},
    {"name": "1-Propanol", "smiles": "CCCO"},
    {"name": "Isopropanol", "smiles": "CC(O)C"},
    {"name": "Acetone", "smiles": "CC(=O)C"},
    {"name": "Acetic acid", "smiles": "CC(=O)O"},
    {"name": "Ethylene glycol", "smiles": "OCCO"},
    {"name": "Glycolaldehyde", "smiles": "OCC=O"},
    {"name": "1,2-Propanediol", "smiles": "CC(O)CO"},
    {"name": "Acetol", "smiles": "CC(=O)CO"},
    {"name": "2-Propanone hydrate", "smiles": "CC(O)(O)C"},
    {"name": "2-Methyl-2-propanol", "smiles": "CC(C)(C)O"},
    {"name": "Allyl alcohol", "smiles": "C=CCO"},
    {"name": "Butanone", "smiles": "CCC(=O)C"},
]

# Illutstrative 2D embedding. Coordinates are hand-placed to make chemical
# families legible; they are not derived from a trained encoder.
DEMO_LATENT: List[Dict[str, object]] = [
    {"name": "Ethanol", "smiles": "CCO", "x": -0.86, "y": 0.52},
    {"name": "Acetone", "smiles": "CC(C)=O", "x": -0.72, "y": 0.24},
    {"name": "Urea", "smiles": "NC(N)=O", "x": -0.82, "y": 0.02},
    {"name": "Imidazole", "smiles": "c1cnc[nH]1", "x": -0.66, "y": -0.22},
    {"name": "Furan", "smiles": "c1ccoc1", "x": -0.54, "y": 0.62},
    {"name": "Pyridine", "smiles": "c1ccncc1", "x": -0.44, "y": 0.34},
    {"name": "Benzene", "smiles": "c1ccccc1", "x": -0.22, "y": 0.10},
    {"name": "Toluene", "smiles": "Cc1ccccc1", "x": -0.06, "y": 0.34},
    {"name": "Phenol", "smiles": "Oc1ccccc1", "x": -0.18, "y": -0.16},
    {"name": "Aniline", "smiles": "Nc1ccccc1", "x": -0.02, "y": -0.38},
    {"name": "Indole", "smiles": "c1ccc2[nH]ccc2c1", "x": 0.16, "y": 0.56},
    {"name": "Quinoline", "smiles": "c1ccc2ncccc2c1", "x": 0.32, "y": 0.78},
    {"name": "Vanillin", "smiles": "COc1cc(C=O)ccc1O", "x": 0.34, "y": 0.04},
    {"name": "Paracetamol", "smiles": "CC(=O)NC1=CC=C(O)C=C1", "x": 0.48, "y": -0.28},
    {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "x": 0.62, "y": -0.08},
    {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "x": 0.72, "y": 0.42},
    {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "x": 0.86, "y": -0.06},
    {"name": "Menthol", "smiles": "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", "x": 0.92, "y": 0.28},
    {"name": "Serotonin", "smiles": "NCCc1c[nH]c2ccc(O)cc12", "x": 0.58, "y": 0.78},
    {"name": "Glucose", "smiles": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "x": 0.44, "y": -0.66},
]

# Small reference set used only to describe the demo batch in the UI.
# It is not a QM9 training split.
DEMO_REFERENCE_SMILES = {
    "CCO",
    "c1ccccc1",
    "CC(C)=O",
    "CC(=O)O",
    "c1ccncc1",
    "Oc1ccccc1",
    "CC(C)CC(=O)O",
    "NC(N)=O",
}
