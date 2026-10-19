"""RDKit chemistry helpers shared by the API.

The 3D embedding path intentionally mirrors the notebook's cheminformatics
tooling (RDKit parsing, sanitisation, canonical SMILES and descriptor
calculation). Coordinates are generated on demand, never faked.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

MAX_ATOMS = 110
ELEMENT_COLORS = {
    "H": "#ffffff",
    "C": "#909090",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "F": "#90e050",
    "P": "#ff8000",
    "S": "#ffff30",
    "Cl": "#1ff01f",
    "Br": "#a62929",
    "I": "#940094",
    "B": "#ffb5b5",
    "Si": "#f0c8a0",
    "Se": "#ffa100",
}


class ChemistryError(ValueError):
    """A user-facing chemistry error."""


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse and sanitise a SMILES string, with friendly errors."""
    if not smiles or not smiles.strip():
        raise ChemistryError("Enter a SMILES string to inspect a molecule.")
    if len(smiles) > 400:
        raise ChemistryError("That SMILES string is too long for this viewer.")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ChemistryError("We couldn't read this SMILES string. Check atom symbols, ring labels and bond syntax.")
    if mol.GetNumHeavyAtoms() == 0:
        raise ChemistryError("That structure contains no heavy atoms.")
    if mol.GetNumAtoms() > MAX_ATOMS:
        raise ChemistryError(f"Molecule is too large for the interactive viewer (limit: {MAX_ATOMS} atoms).")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # pragma: no cover - RDKit exception types vary
        raise ChemistryError("The molecule parsed but failed chemical sanitisation.") from exc
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def molecular_formula(mol: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    """Descriptors aligned with the notebook's property analysis, plus a few
    assessment-friendly extras."""
    return {
        "formula": molecular_formula(mol),
        "molecularWeight": round(float(Descriptors.MolWt(mol)), 2),
        "exactMass": round(float(Descriptors.ExactMolWt(mol)), 4),
        "logp": round(float(Descriptors.MolLogP(mol)), 2),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": round(float(Descriptors.TPSA(mol)), 2),
        "rotatableBonds": int(Descriptors.NumRotatableBonds(mol)),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "heavyAtoms": int(mol.GetNumHeavyAtoms()),
        "qed": round(float(Descriptors.qed(mol)), 3),
    }


def element_summary(mol: Chem.Mol) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"symbol": symbol, "count": count, "color": ELEMENT_COLORS.get(symbol, "#b0b0b0")}
        for symbol, count in ordered
    ]


def embed_3d(mol: Chem.Mol) -> Dict[str, Any]:
    """Generate a real 3D conformer and return a V2000 mol block.

    ETKDGv3 is the same family of distance-geometry embedding used in standard
    RDKit molecular workflows. A random-coordinate retry handles awkward
    topologies, and MMFF94 (or UFF as a fallback) relaxes the geometry.
    """
    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0x5EED
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.enforceChirality = True

    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        retry = AllChem.ETKDGv3()
        retry.randomSeed = 0x5EED + 1
        retry.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol_h, retry)
    if status != 0:
        raise ChemistryError("RDKit could not generate a 3D conformer for this structure.")

    energy: Optional[float] = None
    forces = "MMFF94"
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if props is not None:
            result = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500, mmffVariant="MMFF94")
            if result == 0:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                if ff is not None:
                    energy = round(float(ff.CalcEnergy()), 3)
        else:
            raise ValueError("MMFF parameters unavailable")
    except Exception:
        try:
            forces = "UFF"
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
        except Exception:
            forces = "none"

    conf = mol_h.GetConformer()
    coords = conf.GetPositions()
    span = {
        "x": round(float(coords[:, 0].max() - coords[:, 0].min()), 3),
        "y": round(float(coords[:, 1].max() - coords[:, 1].min()), 3),
        "z": round(float(coords[:, 2].max() - coords[:, 2].min()), 3),
    }
    # Guard against a silently failed embedding (all coordinates at the origin).
    if max(span.values()) < 0.01:
        raise ChemistryError("The generated conformer was degenerate; try another molecule.")

    molblock = Chem.MolToMolBlock(mol_h, confId=0)
    return {
        "molblock": molblock,
        "atoms": int(mol_h.GetNumAtoms()),
        "bonds": int(mol_h.GetNumBonds()),
        "heavyAtoms": int(mol.GetNumHeavyAtoms()),
        "energy": energy,
        "forceField": forces,
        "coordinateSpan": span,
        "has3d": True,
    }


def molecule_payload(smiles: str) -> Dict[str, Any]:
    mol = parse_smiles(smiles)
    canon = canonical_smiles(mol)
    geom = embed_3d(mol)
    return {
        "inputSmiles": smiles.strip(),
        "smiles": canon,
        "valid": True,
        "name": None,
        "descriptors": descriptors(mol),
        "elements": element_summary(mol),
        "geometry": geom,
        "source": "rdkit",
    }


def candidate_payload(smiles: str, name: Optional[str], index: int, source: str) -> Dict[str, Any]:
    """Build a candidate record without paying for a 3D embedding.

    3D is generated lazily when the candidate is selected for viewing.
    """
    try:
        mol = parse_smiles(smiles)
        canon = canonical_smiles(mol)
        desc = descriptors(mol)
        return {
            "id": f"{source}-{index}",
            "name": name,
            "smiles": canon,
            "inputSmiles": smiles,
            "valid": True,
            "descriptors": desc,
            "source": source,
        }
    except ChemistryError:
        return {
            "id": f"{source}-{index}",
            "name": name,
            "smiles": smiles,
            "inputSmiles": smiles,
            "valid": False,
            "descriptors": None,
            "source": source,
        }


def sdf_document(smiles_list: List[str], sources: Optional[List[str]] = None) -> tuple[str, int, int]:
    """Build a multi-record SDF. Returns (sdf, written, skipped)."""
    buffer = io.StringIO()
    writer = Chem.SDWriter(buffer)
    written = 0
    skipped = 0
    for idx, smiles in enumerate(smiles_list, start=1):
        try:
            mol = parse_smiles(smiles)
            geom = embed_3d(mol)
            mol_h = Chem.MolFromMolBlock(geom["molblock"], removeHs=False, sanitize=True)
            if mol_h is None:
                raise ChemistryError("mol block parse failed")
            source = sources[idx - 1] if sources and idx <= len(sources) else "VAE"
            is_reference = source == "related-reference"
            mol_h.SetProp("_Name", f"Candidate {idx}" + (" — related reference" if is_reference else ""))
            mol_h.SetProp("Source", "Related reference" if is_reference else source)
            mol_h.SetProp("SMILES", canonical_smiles(mol))
            mol_h.SetProp("Formula", molecular_formula(mol))
            mol_h.SetProp("MolecularWeight", f"{Descriptors.MolWt(mol):.3f}")
            writer.write(mol_h)
            written += 1
        except Exception:
            skipped += 1
    writer.close()
    return buffer.getvalue(), written, skipped


def demo_batch_metrics(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [c for c in candidates if c["valid"] and c["descriptors"]]
    unique = {c["smiles"] for c in valid}
    avg_mw = sum(c["descriptors"]["molecularWeight"] for c in valid) / len(valid) if valid else 0
    avg_logp = sum(c["descriptors"]["logp"] for c in valid) / len(valid) if valid else 0
    return {
        "total": len(candidates),
        "valid": len(valid),
        "validity": round(len(valid) / len(candidates), 3) if candidates else 0,
        "unique": len(unique),
        "uniqueness": round(len(unique) / len(valid), 3) if valid else 0,
        "averageMolecularWeight": round(avg_mw, 2),
        "averageLogP": round(avg_logp, 2),
    }


def finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None
