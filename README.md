# Molecule Studio

A lightweight React/Vite and FastAPI demo for the notebook’s GRU SMILES VAE. RDKit parses molecules, calculates descriptors and generates optimized 3D conformers; 3Dmol.js renders the interactive viewer.

## Run locally

```sh
# Python 3.10+ and Node.js 20+; install the small web runtime first.
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install

# Terminal 1: API
uvicorn server.main:app --reload

# Terminal 2: UI
npm run dev
```

Open the Vite URL printed in the terminal. The web UI proxies `/api` to `localhost:8000`.

## Model and demo data

The trained weights, config, model definition, and vocabulary are in `trained_vae_artifacts/`. The API loads `vae_model.pt` and its matching vocabulary automatically when PyTorch is installed. The artifact uses an embedding-based GRU VAE, so inference follows its saved architecture and padding/token settings. Checkpoint tensors are loaded on CPU; no GPU or training run is required. `checkpoint.pt` is retained as the full training checkpoint.

If PyTorch is unavailable, install the optional dependency with `pip install -r requirements-model.txt` and restart the API. In that case the app clearly switches to curated demo candidates and an illustrative latent map; it does not label them as model inference. All candidate structures are still validated and embedded in 3D by RDKit.

When a decoded string cannot be displayed as a molecule, the gallery chooses the closest vocabulary-compatible reference by SMILES edit distance. Cards and exports label these as related references. Model outputs and reference structures keep separate metrics.

## Exports

Candidate lists can be downloaded as descriptor CSV or as SDF with RDKit-generated 3D conformers. Inputs RDKit cannot parse are recorded in CSV and omitted from SDF with a skipped-record count.
