"""FastAPI endpoints for Molecule Studio's RDKit and optional notebook VAE."""
from __future__ import annotations

import csv
import io
import random
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chem import ChemistryError, candidate_payload, demo_batch_metrics, molecule_payload, parse_smiles, descriptors, canonical_smiles, sdf_document
from .demo_data import DEMO_LATENT, DEMO_MOLECULES, RELATED_MOLECULES
from .vae import ModelRuntime

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent / "trained_vae_artifacts"
runtime = ModelRuntime(ARTIFACTS / "vae_model.pt", ARTIFACTS / "vocab.json")
app = FastAPI(title="Molecule Studio API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SmilesInput(BaseModel):
    smiles: str = Field(max_length=400)


class GenerateInput(BaseModel):
    count: int = Field(default=6, ge=1, le=20)
    temperature: float = Field(default=0.65, ge=0.2, le=2.0)


class InterpolationInput(BaseModel):
    smilesA: str
    smilesB: str
    steps: int = Field(default=7, ge=3, le=15)


def prepare_candidates(raw_smiles: List[str], source: str):
    """Keep raw-model metrics separate while supplying explorable structures."""
    raw_candidates = [candidate_payload(s, None, i, source) for i, s in enumerate(raw_smiles)]
    model_metrics = demo_batch_metrics(raw_candidates)
    candidates = []
    related_count = 0
    used_references = set()
    for index, (smiles, raw_candidate) in enumerate(zip(raw_smiles, raw_candidates)):
        if source == "vae" and not raw_candidate["valid"]:
            example = nearest_related_molecule(smiles, used_references)
            used_references.add(example["smiles"])
            candidate = candidate_payload(example["smiles"], example["name"], index, "related-reference")
            candidate["id"] = f"related-reference-{index}"
            candidate["relatedReference"] = True
            candidate["originalOutput"] = smiles
            related_count += 1
        else:
            candidate = raw_candidate
            candidate["id"] = f"{source}-{index}"
            candidate["relatedReference"] = False
            candidate["originalOutput"] = smiles
        candidates.append(candidate)
    return {
        "candidates": candidates,
        "metrics": demo_batch_metrics(candidates),
        "modelMetrics": model_metrics,
        "source": source,
        "relatedCount": related_count,
    }


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, start=1):
        current = [row]
        for column, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def nearest_related_molecule(smiles: str, excluded: set[str] | None = None):
    """Find the closest valid molecule supported by the checkpoint vocabulary."""
    excluded = excluded or set()
    choices = [m for m in RELATED_MOLECULES if m["smiles"] not in excluded] or RELATED_MOLECULES
    return min(choices, key=lambda molecule: (_edit_distance(smiles, molecule["smiles"]), abs(len(smiles) - len(molecule["smiles"]))))


@app.get("/api/status")
def status():
    return {"mode": "vae" if runtime.loaded else "demo", "model": runtime.status()}


@app.post("/api/molecule")
def inspect_molecule(payload: SmilesInput):
    try:
        return molecule_payload(payload.smiles)
    except ChemistryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/geometry")
def geometry(payload: SmilesInput):
    try:
        return molecule_payload(payload.smiles)["geometry"]
    except ChemistryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/generate")
def generate(payload: GenerateInput):
    if runtime.loaded:
        raw_smiles = runtime.generate(payload.count, temperature=payload.temperature)
        source = "vae"
    else:
        # Stable, curated selections make the demo reproducible and candid.
        seed = int(payload.temperature * 1000) + payload.count * 100
        selected = random.Random(seed).sample(DEMO_MOLECULES, min(payload.count, len(DEMO_MOLECULES)))
        raw_smiles = [item["smiles"] for item in selected]
        source = "demo"
    return prepare_candidates(raw_smiles, source)


@app.get("/api/latent")
def latent():
    if runtime.loaded:
        # The artifact vocabulary is intentionally small. Project encodable,
        # chemically valid examples with the trained encoder instead of
        # showing unsupported demo molecules as if the model had encoded them.
        examples = [
            ("Methane", "C"), ("Ethane", "CC"), ("Ethylene", "C=C"),
            ("Propane", "CCC"), ("Formaldehyde", "C=O"),
            ("Ethanol", "CCO"), ("1-Propanol", "CCCO"),
            ("Acetone", "CC(C)=O"), ("Acetic acid", "CC(=O)O"),
            ("Isopropanol", "CC(O)C"),
        ]
        encodable = [(name, smiles) for name, smiles in examples if set(smiles) <= set(runtime.vocab)]
        vectors = runtime.encode([smiles for _, smiles in encodable]).detach().cpu().numpy()
        projected = runtime.project_2d(vectors)
        points = [{"name": name, "smiles": smiles, "x": float(projected[i, 0]), "y": float(projected[i, 1])}
                  for i, (name, smiles) in enumerate(encodable)]
        return {"points": points, "illustrative": False, "source": "vae", "projection": "PCA"}
    return {"points": DEMO_LATENT, "illustrative": True}


@app.post("/api/interpolate")
def interpolate(payload: InterpolationInput):
    try:
        mol_a = parse_smiles(payload.smilesA)
        mol_b = parse_smiles(payload.smilesB)
    except ChemistryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if runtime.loaded:
        supported = set(runtime.vocab) - {"<", ">"}
        unsupported = sorted((set(canonical_smiles(mol_a)) | set(canonical_smiles(mol_b))) - supported)
        if unsupported:
            raise HTTPException(status_code=422, detail=f"This checkpoint's SMILES vocabulary does not support: {', '.join(unsupported)}.")
        try:
            decoded, _ = runtime.interpolation(canonical_smiles(mol_a), canonical_smiles(mol_b), payload.steps)
            steps = []
            for i, smiles in enumerate(decoded):
                try:
                    mol = parse_smiles(smiles)
                    smiles = canonical_smiles(mol)
                    valid = True
                except ChemistryError:
                    valid = False
                steps.append({"alpha": round(i / max(1, len(decoded) - 1), 3), "smiles": smiles, "valid": valid})
            return {"steps": steps, "source": "vae"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VAE interpolation failed: {exc}") from exc

    # In demo mode interpolate along the illustrative 2D point map, selecting
    # real curated molecules nearest each linearly interpolated position.
    points = DEMO_LATENT
    by_smiles = {canonical_smiles(parse_smiles(str(p["smiles"]))): p for p in points}
    a = by_smiles.get(canonical_smiles(mol_a))
    b = by_smiles.get(canonical_smiles(mol_b))
    if a is None:
        a = points[0]
    if b is None:
        b = points[-1]
    steps = []
    used = set()
    for i in range(payload.steps):
        alpha = i / max(1, payload.steps - 1)
        x = float(a["x"]) * (1-alpha) + float(b["x"]) * alpha
        y = float(a["y"]) * (1-alpha) + float(b["y"]) * alpha
        point = min(points, key=lambda p: (float(p["x"])-x)**2 + (float(p["y"])-y)**2 + (0.02 if p["smiles"] in used else 0))
        used.add(str(point["smiles"]))
        steps.append({"alpha": round(alpha, 3), "smiles": point["smiles"], "valid": True})
    return {"steps": steps, "source": "illustrative"}


@app.get("/api/export/{fmt}")
def export(fmt: str, smiles: str, sources: str = ""):
    items = [s.strip() for s in smiles.splitlines() if s.strip()]
    source_items = [s.strip() for s in sources.split(",") if s.strip()]
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["smiles", "formula", "molecular_weight", "exact_mass", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "rings", "valid", "source"])
        for index, item in enumerate(items):
            source = source_items[index] if index < len(source_items) else "VAE"
            try:
                mol = parse_smiles(item)
                d = descriptors(mol)
                writer.writerow([canonical_smiles(mol), d["formula"], d["molecularWeight"], d["exactMass"], d["logp"], d["hbd"], d["hba"], d["tpsa"], d["rotatableBonds"], d["rings"], True, "Related reference" if source == "related-reference" else source])
            except ChemistryError:
                writer.writerow([item, "", "", "", "", "", "", "", "", "", False, source])
        return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=molecule-results.csv"})
    if fmt == "sdf":
        sdf, written, skipped = sdf_document(items, source_items)
        if not written:
            raise HTTPException(status_code=422, detail="No valid molecules could be embedded for SDF export.")
        return StreamingResponse(iter([sdf]), media_type="chemical/x-mdl-sdfile", headers={"Content-Disposition": "attachment; filename=molecule-results.sdf", "X-Export-Skipped": str(skipped)})
    raise HTTPException(status_code=404, detail="Choose csv or sdf.")


@app.get("/api/health")
def health():
    return {"ok": True}
