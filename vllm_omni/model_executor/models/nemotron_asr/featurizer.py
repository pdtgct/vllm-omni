# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NeMo-exact mel featurizer (PORT-FEAT-001).

Reproduces NeMo ``FilterbankFeatures`` inference behavior at the pinned
commit (features.py:378-473 @ de242add): masked preemphasis, constant-pad
centered STFT, power-2 magnitude, mel projection, ``log(x + guard)``.
Normalization is "NA" for this checkpoint and dither is train-only, so
neither appears here. The mel filterbank arrives as a tensor from the
offline conversion tool (NeMo computes it with librosa at graph build;
shipping the computed bank removes the runtime dependency and the
reimplementation risk — parity is asserted against a NeMo-produced
fixture).
"""

import torch
from torch import nn


class MelFeaturizer(nn.Module):
    """Log-mel features over a whole buffer, one call per utterance/chunk.

    Streaming continuity (raw-sample tail, pre-encode overlap,
    ``drop_extra_pre_encoded``) is the serving layer's job
    (PORT-FEAT-002); this module is stateless.
    """

    def __init__(
        self,
        *,
        filterbank: torch.Tensor,
        window: torch.Tensor,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        preemph: float = 0.97,
        log_zero_guard: float = 2.0**-24,
    ) -> None:
        super().__init__()
        if filterbank.dim() != 2 or filterbank.shape[1] != n_fft // 2 + 1:
            raise ValueError(f"filterbank must be (n_mels, {n_fft // 2 + 1}), got {tuple(filterbank.shape)}")
        if window.shape != (win_length,):
            raise ValueError(f"window must be ({win_length},), got {tuple(window.shape)}")
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemph = preemph
        self.log_zero_guard = log_zero_guard
        self.register_buffer("fb", filterbank.to(torch.float32))
        # The checkpoint's persisted window, like the filterbank — a
        # freshly built hann differs by one ulp (trained under a
        # different torch rounding), which log() amplifies to ~1e-4 at
        # quiet mel bins. Checkpoint buffers are inputs, not recomputed.
        self.register_buffer("window", window.to(torch.float32))

    def output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        """Mel frame count per NeMo ``get_seq_len`` (center=True)."""
        pad = 2 * (self.n_fft // 2)
        return torch.div(
            sample_lengths + pad - self.n_fft,
            self.hop_length,
            rounding_mode="floor",
        ).to(torch.long)

    def forward(self, waveform: torch.Tensor, sample_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-mel features.

        Args:
            waveform: ``(batch, samples)`` fp32 PCM in [-1, 1].
            sample_lengths: valid samples per batch element.

        Returns:
            ``(batch, n_mels, frames)`` log-mel and per-element frame
            counts.
        """
        mel_len = self.output_lengths(sample_lengths)

        # Masked preemphasis: first sample kept, x[t] - p*x[t-1], zeros
        # past each element's valid length (features.py:431-434).
        time_mask = torch.arange(waveform.shape[1], device=waveform.device).unsqueeze(0) < sample_lengths.unsqueeze(1)
        x = torch.cat(
            (
                waveform[:, :1],
                waveform[:, 1:] - self.preemph * waveform[:, :-1],
            ),
            dim=1,
        )
        x = x.masked_fill(~time_mask, 0.0)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
        )
        # NeMo computes magnitude (sqrt) then squares it back to power
        # (features.py:443-453). Algebraically an identity, numerically
        # an fp32 round-trip that log() then amplifies at low-energy
        # bins — parity requires the same op order.
        magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1))
        power = magnitude.pow(2.0)
        mel = torch.matmul(self.fb, power)
        return torch.log(mel + self.log_zero_guard), mel_len
