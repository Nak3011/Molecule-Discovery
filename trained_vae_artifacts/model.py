import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_dim=384, latent_dim=128):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.pad_id = vocab_size

        # The extra embedding index is reserved for padding.
        self.embedding = nn.Embedding(
            vocab_size + 1, 128, padding_idx=self.pad_id
        )

        self.encoder = nn.GRU(
            128, hidden_dim, batch_first=True
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_embedding = nn.Embedding(
            vocab_size + 1, 128, padding_idx=self.pad_id
        )

        self.decoder = nn.GRU(
            128 + latent_dim, hidden_dim, batch_first=True
        )

        self.fc_output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, tokens, lengths):
        embedded = self.embedding(tokens)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h = self.encoder(packed)
        h = h[-1]

        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_teacher_forced(self, tokens, z):
        # Predict token t+1 from the input token at t.
        decoder_tokens = tokens[:, :-1]
        embedded = self.decoder_embedding(decoder_tokens)

        z_sequence = z.unsqueeze(1).expand(
            -1, embedded.size(1), -1
        )

        decoder_input = torch.cat(
            [embedded, z_sequence], dim=-1
        )

        output, _ = self.decoder(decoder_input)
        return self.fc_output(output)

    def forward(self, tokens, lengths):
        mu, logvar = self.encode(tokens, lengths)
        z = self.reparameterize(mu, logvar)

        logits = self.decode_teacher_forced(tokens, z)
        targets = tokens[:, 1:]

        return logits, targets, mu, logvar

    def decode_step(self, token, z, h):
        embedded = self.decoder_embedding(token).unsqueeze(1)
        decoder_input = torch.cat(
            [embedded, z.unsqueeze(1)], dim=-1
        )

        output, h = self.decoder(decoder_input, h)
        logits = self.fc_output(output[:, 0, :])

        return logits, h
