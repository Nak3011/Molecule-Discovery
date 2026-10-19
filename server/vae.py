"""Optional runtime for the notebook's SMILES VAE.

The architecture, vocabulary handling and decoding loop below are adapted from
``Variational_Autoencoder_for_Molecule_Discovery.ipynb`` so a checkpoint trained
in the notebook can be dropped in without reimplementation.

The website runs in clearly-labelled demo mode when no checkpoint (or no
PyTorch installation) is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VAE:
    """Lazy import wrapper so the base demo does not require PyTorch."""

    @staticmethod
    def build(torch: Any, nn: Any, vocab_size: int, hidden_dim: int, latent_dim: int):
        class _VAE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_dim = hidden_dim
                self.latent_dim = latent_dim
                self.encoder = nn.GRU(vocab_size, hidden_dim, batch_first=True)
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.GRU(vocab_size + latent_dim, hidden_dim, batch_first=True)
                self.fc_output = nn.Linear(hidden_dim, vocab_size)

            def encode(self, x):
                _, h = self.encoder(x)
                h = h.squeeze(0)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z, max_length, target=None):
                batch_size = z.size(0)
                h = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
                x = torch.zeros(batch_size, 1, self.vocab_size).to(z.device)
                outputs = []
                for _ in range(max_length):
                    decoder_input = torch.cat([x, z.unsqueeze(1)], dim=2)
                    output, h = self.decoder(decoder_input, h)
                    output = self.fc_output(output)
                    outputs.append(output)
                    if target is not None:
                        x = target[:, _ : _ + 1, :]
                    else:
                        x = torch.softmax(output, dim=-1)
                return torch.cat(outputs, dim=1)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z, x.size(1), target=x), mu, logvar

        return _VAE()

    @staticmethod
    def build_artifact(torch: Any, nn: Any, vocab_size: int, hidden_dim: int, latent_dim: int, embedding_dim: int):
        """Build the embedding-based architecture used by trained_vae_artifacts."""
        from torch.nn.utils.rnn import pack_padded_sequence

        class _ArtifactVAE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_dim = hidden_dim
                self.latent_dim = latent_dim
                self.pad_id = vocab_size
                self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=self.pad_id)
                self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
                self.decoder_embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=self.pad_id)
                self.decoder = nn.GRU(embedding_dim + latent_dim, hidden_dim, batch_first=True)
                self.fc_output = nn.Linear(hidden_dim, vocab_size)

            def encode(self, tokens, lengths):
                embedded = self.embedding(tokens)
                packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, hidden = self.encoder(packed)
                hidden = hidden[-1]
                return self.fc_mu(hidden), self.fc_logvar(hidden)

            def decode_step(self, token, z, hidden):
                embedded = self.decoder_embedding(token).unsqueeze(1)
                decoder_input = torch.cat([embedded, z.unsqueeze(1)], dim=-1)
                output, hidden = self.decoder(decoder_input, hidden)
                return self.fc_output(output[:, 0, :]), hidden

        return _ArtifactVAE()


def create_vocabulary(smiles_list: List[str]) -> Dict[str, int]:
    vocab = {"<": 0, ">": 1}
    for smiles in smiles_list:
        for char in smiles:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def smiles_to_onehot(torch: Any, smiles: str, vocab: Dict[str, int], max_length: int = 100):
    s = "<" + smiles + ">"
    encoded = torch.zeros(max_length, len(vocab))
    for i, c in enumerate(s[:max_length]):
        encoded[i, vocab.get(c, 0)] = 1
    return encoded


def decode_from_latent(
    torch: Any,
    model: Any,
    z: Any,
    vocab: Dict[str, int],
    max_length: int = 100,
    temperature: float = 1.0,
    greedy: bool = False,
) -> List[str]:
    """The notebook's shared decoding routine (sampling, reconstruction, interpolation)."""
    model.eval()
    device = next(model.parameters()).device
    inv_vocab = {v: k for k, v in vocab.items()}
    skip_tokens = {vocab["<"], vocab[">"]}

    z = z.to(device)
    batch = z.size(0)
    results = ["" for _ in range(batch)]
    finished = [False] * batch

    with torch.no_grad():
        x = torch.zeros(batch, 1, model.vocab_size, device=device)
        x[:, 0, vocab["<"]] = 1
        h = torch.zeros(1, batch, model.hidden_dim, device=device)

        for _ in range(max_length):
            decoder_input = torch.cat([x, z.unsqueeze(1)], dim=2)
            output, h = model.decoder(decoder_input, h)
            logits = model.fc_output(output).squeeze(1) / temperature

            if greedy:
                next_chars = logits.argmax(dim=-1)
            else:
                next_chars = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)

            x = torch.zeros(batch, 1, model.vocab_size, device=device)
            for b in range(batch):
                idx = int(next_chars[b])
                if not finished[b]:
                    if idx == vocab[">"]:
                        finished[b] = True
                    elif idx not in skip_tokens:
                        results[b] += inv_vocab.get(idx, "")
                x[b, 0, idx] = 1
            if all(finished):
                break

    return results


class ModelRuntime:
    """Loads an optional notebook checkpoint and exposes generation helpers."""

    def __init__(self, checkpoint_path: Path, vocab_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.loaded = False
        self.reason = "No trained checkpoint bundled; demo mode is active."
        self.torch: Any = None
        self.model: Any = None
        self.vocab: Dict[str, int] = {}
        self.max_length = 100
        self.latent_dim = 128
        self.hidden_dim = 512
        self.embedding_dim = 128
        self.artifact_architecture = False
        self._load()

    def _read_vocab(self, payload: Any) -> Optional[Dict[str, int]]:
        vocab = None
        if isinstance(payload, dict):
            vocab = payload.get("vocab") or payload.get("vocabulary")
        if vocab is None and self.vocab_path.exists():
            with self.vocab_path.open("r", encoding="utf-8") as handle:
                vocab = json.load(handle)
        if isinstance(vocab, list):
            return {str(token): idx for idx, token in enumerate(vocab)}
        if isinstance(vocab, dict):
            return {str(token): int(idx) for token, idx in vocab.items()}
        return None

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self.reason = "Checkpoint found, but PyTorch is not installed. Demo mode is active."
            return

        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(self.checkpoint_path, map_location="cpu")
        except Exception as exc:
            self.reason = f"Checkpoint could not be read ({exc}). Demo mode is active."
            return

        state = payload
        config: Dict[str, Any] = {}
        if isinstance(payload, dict) and any(key in payload for key in ("state_dict", "model_state_dict")):
            state = payload.get("state_dict") or payload.get("model_state_dict")
            config = payload.get("config") or payload
        config_path = self.checkpoint_path.parent / "config.json"
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    config = {**json.load(handle), **(config if isinstance(config, dict) else {})}
            except (OSError, ValueError):
                pass
        if not isinstance(state, dict):
            self.reason = "Checkpoint format was not recognised. Demo mode is active."
            return

        vocab = self._read_vocab(payload)
        if not vocab:
            self.reason = "Checkpoint found, but the SMILES vocabulary is missing. Demo mode is active."
            return

        try:
            latent_dim = int(state["fc_mu.weight"].shape[0])
            hidden_dim = int(state["fc_mu.weight"].shape[1])
            vocab_size = int(state["fc_output.weight"].shape[0])
            if "embedding.weight" in state:
                embedding_dim = int(state["embedding.weight"].shape[1])
                model = VAE.build_artifact(torch, nn, vocab_size, hidden_dim, latent_dim, embedding_dim)
                self.artifact_architecture = True
                self.embedding_dim = embedding_dim
            else:
                model = VAE.build(torch, nn, vocab_size, hidden_dim, latent_dim)
            model.load_state_dict(state)
            model.eval()
        except Exception as exc:
            self.reason = f"Checkpoint could not be loaded into the VAE ({exc}). Demo mode is active."
            return

        if len(vocab) != vocab_size:
            self.reason = "Vocabulary size does not match the checkpoint. Demo mode is active."
            return

        self.torch = torch
        self.model = model
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embedding_dim = int(config.get("embedding_dim", self.embedding_dim))
        self.latent_dim = latent_dim
        self.max_length = int(config.get("max_length", 100)) if isinstance(config, dict) else 100
        self.loaded = True
        self.reason = "Trained VAE artifact loaded from trained_vae_artifacts."

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "reason": self.reason,
            "checkpoint": str(self.checkpoint_path),
            "latentDimension": self.latent_dim if self.loaded else None,
            "architecture": "embedded_gru_vae" if self.loaded and self.artifact_architecture else "notebook_gru_vae" if self.loaded else None,
        }

    def _artifact_tokens(self, smiles_list: List[str]):
        torch = self.torch
        token_ids = []
        lengths = []
        start_id = self.vocab["<"]
        end_id = self.vocab[">"]
        for smiles in smiles_list:
            sequence = [start_id] + [self.vocab.get(char, start_id) for char in smiles] + [end_id]
            sequence = sequence[: self.max_length]
            lengths.append(len(sequence))
            sequence += [self.model.pad_id] * (self.max_length - len(sequence))
            token_ids.append(sequence)
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    def _decode_artifact(self, z: Any, temperature: float = 1.0, greedy: bool = False) -> List[str]:
        torch = self.torch
        device = next(self.model.parameters()).device
        z = z.to(device)
        batch = z.size(0)
        token = torch.full((batch,), self.vocab["<"], dtype=torch.long, device=device)
        hidden = torch.zeros(1, batch, self.model.hidden_dim, device=device)
        inv_vocab = {index: char for char, index in self.vocab.items()}
        results = [""] * batch
        finished = [False] * batch
        with torch.no_grad():
            for _ in range(self.max_length):
                logits, hidden = self.model.decode_step(token, z, hidden)
                logits = logits / max(float(temperature), 1e-4)
                if greedy:
                    next_tokens = logits.argmax(dim=-1)
                else:
                    next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
                token = next_tokens
                for index, raw_token in enumerate(next_tokens.tolist()):
                    if not finished[index]:
                        if raw_token == self.vocab[">"]:
                            finished[index] = True
                        elif raw_token != self.vocab["<"]:
                            results[index] += inv_vocab.get(raw_token, "")
                if all(finished):
                    break
        return results

    def generate(self, count: int, seed: Optional[int] = None, temperature: float = 0.65) -> List[str]:
        if not self.loaded:
            raise RuntimeError("No trained checkpoint is loaded.")
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(count, self.model.latent_dim)
        if self.artifact_architecture:
            return self._decode_artifact(z, temperature=temperature, greedy=False)
        return decode_from_latent(torch, self.model, z, self.vocab, max_length=self.max_length, temperature=temperature, greedy=False)

    def encode(self, smiles_list: List[str]) -> Any:
        if not self.loaded:
            raise RuntimeError("No trained checkpoint is loaded.")
        torch = self.torch
        if self.artifact_architecture:
            tokens, lengths = self._artifact_tokens(smiles_list)
            with torch.no_grad():
                mu, _ = self.model.encode(tokens, lengths)
        else:
            tensors = [smiles_to_onehot(torch, smiles, self.vocab, max_length=self.max_length) for smiles in smiles_list]
            x = torch.stack(tensors)
            with torch.no_grad():
                mu, _ = self.model.encode(x)
        return mu

    def interpolation(self, smiles_a: str, smiles_b: str, steps: int = 7) -> Tuple[List[str], Any]:
        """Encode endpoints, walk the latent line, and decode each point (greedy)."""
        if not self.loaded:
            raise RuntimeError("No trained checkpoint is loaded.")
        torch = self.torch
        mu = self.encode([smiles_a, smiles_b])
        z_a, z_b = mu[0], mu[1]
        alphas = torch.linspace(0, 1, steps)
        z_path = (1 - alphas).unsqueeze(1) * z_a.unsqueeze(0) + alphas.unsqueeze(1) * z_b.unsqueeze(0)
        if self.artifact_architecture:
            decoded = self._decode_artifact(z_path, temperature=1.0, greedy=True)
        else:
            decoded = decode_from_latent(torch, self.model, z_path, self.vocab, max_length=self.max_length, temperature=1.0, greedy=True)
        return decoded, z_path.detach().cpu().numpy()

    @staticmethod
    def project_2d(vectors: np.ndarray) -> np.ndarray:
        """Simple PCA projection for an illustrative latent map."""
        if vectors.shape[0] == 0:
            return np.zeros((0, 2))
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        if vectors.shape[0] < 2:
            return np.zeros((vectors.shape[0], 2))
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt[:2].T
        if projected.shape[0] == 1:
            return np.zeros((1, 2))
        scale = np.abs(projected).max()
        if scale > 0:
            projected = projected / scale
        return projected
