"""
Wavelet-based pipeline to convert a 1D time series window into an RGB tensor [3, H, W].

This module provides a composable, testable set of classes:
- WaveletConfig: configuration and validation
- ScalesComputer: precomputes wavelet scales and frequency grid
- SignalWindowExtractor: extracts an exact-duration window with padding policies
- CWTTransformer: computes CWT coefficients using pywt
- SpectrogramPostProcessor: crops in frequency, applies magnitude/clamp/normalization, and resizes
- RGBTensorExporter: maps scalar spectrogram to RGB via a colormap LUT and returns CHW
- CWTImagePipeline: orchestrates the above for simple usage

All components operate primarily on numpy arrays. If return_torch=True, the final
export returns a torch.Tensor of dtype float32 and shape [3, H, W].

Dependencies:
  - numpy, pywt
  - scipy.ndimage (optional) for resizing; if unavailable and return_torch=True,
    torch.nn.functional.interpolate is used as a fallback.
  - matplotlib (only for colormap LUT; no figure rendering is performed)

Example usage inside a PyTorch Dataset:

    from .wavelets import WaveletConfig, CWTImagePipeline

    class MyDataset(Dataset):
        def __init__(self, ...):
            self.pipeline = CWTImagePipeline(
                WaveletConfig(
                    sample_rate_hz=5000,
                    freq_min_hz=0.1,
                    freq_max_hz=20.0,
                    freq_step_hz=0.1,
                    time_window_s=1.0,
                    output_size=244,
                    clamp_min=-0.05,
                    clamp_max=0.05,
                    cmap_name="coolwarm",
                    return_torch=True,
                    magnitude_mode="real",
                    pad_mode="constant",
                )
            )

        def __getitem__(self, idx):
            signal = ... # 1D numpy array or torch tensor
            start_idx = 0
            rgb = self.pipeline(signal, start_idx)
            return rgb, label
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    F = None
    _TORCH_AVAILABLE = False

try:
    from scipy import ndimage as _ndimage
    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _ndimage = None
    _SCIPY_AVAILABLE = False

import pywt

try:
    from matplotlib import cm as _cm
    _MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    _cm = None
    _MATPLOTLIB_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class WaveletConfig:
    """Configuration for CWT image generation.

    Attributes:
        wavelet_name: Name of the continuous wavelet (e.g., "morl").
        sample_rate_hz: Sampling rate of the input 1D signal in Hz.
        freq_min_hz: Minimum frequency to include (after CWT) in Hz.
        freq_max_hz: Maximum frequency to include (after CWT) in Hz.
        freq_step_hz: Frequency grid step used to derive scales.
        time_window_s: Duration of the window to extract, in seconds.
        output_size: Target height/width for the square output spectrogram.
        clamp_min: Lower clamp bound for real-valued coefficient display mode.
        clamp_max: Upper clamp bound for real-valued coefficient display mode.
        cmap_name: Matplotlib colormap name for RGB mapping.
        return_torch: If True, exporter returns a torch.FloatTensor [3, H, W].
        magnitude_mode: One of {"real", "abs", "power"}.
        pad_mode: Padding policy when window exceeds signal bounds; one of
                  {"constant", "reflect", "edge"}.
    """

    sample_rate_hz: float
    wavelet_name: str = "morl"
    freq_min_hz: float = 0.0
    freq_max_hz: float = 20.0
    freq_step_hz: float = 0.1
    time_window_s: float = 1.0
    output_size: int = 244
    clamp_min: float = -0.05
    clamp_max: float = 0.05
    cmap_name: str = "coolwarm"
    return_torch: bool = False
    magnitude_mode: str = "real"  # "real", "abs", or "power"
    pad_mode: str = "constant"    # "constant", "reflect", "edge"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.freq_step_hz <= 0:
            raise ValueError("freq_step_hz must be positive")
        if self.freq_max_hz <= 0:
            raise ValueError("freq_max_hz must be positive")
        if self.freq_min_hz < 0 or self.freq_min_hz >= self.freq_max_hz:
            raise ValueError("freq_min_hz must be >=0 and < freq_max_hz")
        if self.time_window_s <= 0:
            raise ValueError("time_window_s must be positive")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.clamp_min >= self.clamp_max:
            raise ValueError("clamp_min must be < clamp_max")
        if self.magnitude_mode not in {"real", "abs", "power"}:
            raise ValueError("magnitude_mode must be one of {'real','abs','power'}")
        if self.pad_mode not in {"constant", "reflect", "edge"}:
            raise ValueError("pad_mode must be one of {'constant','reflect','edge'}")

    @property
    def n_window_samples(self) -> int:
        return int(round(self.time_window_s * float(self.sample_rate_hz)))


# -----------------------------
# Utilities
# -----------------------------


class ScalesComputer:
    """Computes and caches scales and frequency grid for a given config."""

    def __init__(self, config: WaveletConfig) -> None:
        self.config = config
        self._wavelet = pywt.ContinuousWavelet(config.wavelet_name)
        self._scales, self._freqs_hz = self._compute()

    def _compute(self) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        f_grid = np.arange(c.freq_min_hz, c.freq_max_hz, c.freq_step_hz, dtype=np.float64)
        if f_grid.size == 0:
            # ensure at least one frequency at freq_max_hz if range collapsed
            f_grid = np.array([c.freq_max_hz], dtype=np.float64)
        sampling_period = 1.0 / float(c.sample_rate_hz)
        # pywt.frequency2scale expects frequencies in Hz corresponding to wavelet central frequency
        scales = pywt.frequency2scale(self._wavelet, f_grid) / sampling_period
        return scales.astype(np.float64), f_grid.astype(np.float64)

    @property
    def wavelet(self) -> pywt.ContinuousWavelet:
        return self._wavelet

    @property
    def scales(self) -> np.ndarray:
        return self._scales

    @property
    def freqs_hz(self) -> np.ndarray:
        return self._freqs_hz


class SignalWindowExtractor:
    """Extracts a fixed-size window from a 1D signal with padding policy."""

    def __init__(self, config: WaveletConfig) -> None:
        self.config = config

    def extract(self, signal: np.ndarray | "torch.Tensor", *, start_idx: Optional[int] = 0,
                center_idx: Optional[int] = None) -> np.ndarray:
        if signal is None:
            raise ValueError("signal cannot be None")
        # Convert torch tensor to numpy if needed
        if _TORCH_AVAILABLE and isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = np.asarray(signal)

        if signal_np.ndim != 1:
            raise ValueError("signal must be 1D")

        n = self.config.n_window_samples
        if center_idx is not None:
            start = int(round(center_idx - n / 2))
        else:
            start = int(start_idx or 0)
        end = start + n

        if start >= 0 and end <= signal_np.shape[0]:
            window = signal_np[start:end]
        else:
            # Apply padding as needed
            pad_left = max(0, -start)
            pad_right = max(0, end - signal_np.shape[0])
            sl_start = max(0, start)
            sl_end = min(signal_np.shape[0], end)
            core = signal_np[sl_start:sl_end]
            if self.config.pad_mode == "constant":
                window = np.pad(core, (pad_left, pad_right), mode="constant")
            else:
                # reflect/edge need the entire signal then slice
                pre = []
                post = []
                if pad_left > 0:
                    pre = np.pad(signal_np[:sl_start], (pad_left - sl_start, 0), mode=self.config.pad_mode)[-pad_left:]
                if pad_right > 0:
                    post = np.pad(signal_np[sl_end:], (0, pad_right - (signal_np.shape[0] - sl_end)), mode=self.config.pad_mode)[:pad_right]
                window = np.concatenate([pre, core, post])

        window = window.astype(np.float32, copy=False)
        if window.shape[0] != n:
            # As a last resort, trim or pad to exact length
            if window.shape[0] > n:
                window = window[:n]
            else:
                window = np.pad(window, (0, n - window.shape[0]), mode="constant")
        return window


class CWTTransformer:
    """Computes Continuous Wavelet Transform coefficients."""

    def __init__(self, config: WaveletConfig, scales_computer: ScalesComputer) -> None:
        self.config = config
        self.scales_computer = scales_computer

    def compute(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if window.ndim != 1:
            raise ValueError("window must be 1D")
        scales = self.scales_computer.scales
        w = self.scales_computer.wavelet
        coef, freqs = pywt.cwt(window, scales, w)
        # pywt returns frequency in Hz relative to sampling period embedded in scales; we built scales with sampling_period, so freqs maps back to our f_grid
        # Ensure orientation: low frequency at bottom, high at top (flipud to match notebook style that used np.flip on axis 0)
        coef = np.flip(coef, axis=0)
        freqs_hz = np.flip(self.scales_computer.freqs_hz, axis=0)
        coef = coef.astype(np.float32, copy=False)
        freqs_hz = freqs_hz.astype(np.float32, copy=False)
        return coef, freqs_hz


class SpectrogramPostProcessor:
    """Post-process CWT coefficients into a normalized square spectrogram [H,W] in [0,1]."""

    def __init__(self, config: WaveletConfig) -> None:
        self.config = config

    def _magnitude(self, coef: np.ndarray) -> np.ndarray:
        mode = self.config.magnitude_mode
        if mode == "real":
            x = np.real(coef)
            x = np.clip(x, self.config.clamp_min, self.config.clamp_max)
            # Map clamp range to [0,1]
            denom = (self.config.clamp_max - self.config.clamp_min)
            x01 = (x - self.config.clamp_min) / denom
            return np.clip(x01, 0.0, 1.0)
        elif mode == "abs":
            x = np.abs(coef)
        elif mode == "power":
            x = np.abs(coef) ** 2
        else:  # pragma: no cover
            raise ValueError("Unsupported magnitude_mode")

        # Robust percentile normalization to [0,1]
        lo, hi = np.percentile(x, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1e-6
        x01 = (x - lo) / (hi - lo)
        return np.clip(x01, 0.0, 1.0)

    def _freq_crop(self, spec: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        c = self.config
        # freqs_hz is ascending after flip. Find indices within bounds.
        idx = np.where((freqs_hz >= c.freq_min_hz) & (freqs_hz <= c.freq_max_hz))[0]
        if idx.size == 0:
            # If no overlap, return the original spec
            return spec
        return spec[idx, :]

    def _resize(self, spec01: np.ndarray) -> np.ndarray:
        Ht, Wt = spec01.shape
        out = self.config.output_size
        if Ht == out and Wt == out:
            return spec01
        if _SCIPY_AVAILABLE:
            zoom_y = out / float(Ht)
            zoom_x = out / float(Wt)
            return _ndimage.zoom(spec01, (zoom_y, zoom_x), order=1).astype(np.float32, copy=False)
        # Torch fallback if allowed
        if self.config.return_torch and _TORCH_AVAILABLE:
            t = torch.from_numpy(spec01).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
            t = F.interpolate(t, size=(out, out), mode="bilinear", align_corners=False)
            return t.squeeze(0).squeeze(0).cpu().numpy()
        # Numpy nearest-neighbor fallback
        y_idx = (np.linspace(0, Ht - 1, out)).astype(np.int32)
        x_idx = (np.linspace(0, Wt - 1, out)).astype(np.int32)
        return spec01[np.ix_(y_idx, x_idx)].astype(np.float32, copy=False)

    def process(self, coef: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        mag01 = self._magnitude(coef)
        mag01 = self._freq_crop(mag01, freqs_hz)
        mag01 = self._resize(mag01)
        return mag01.astype(np.float32, copy=False)


class RGBTensorExporter:
    """Converts a [H,W] float32 in [0,1] to an RGB [3,H,W] tensor via LUT colormap."""

    def __init__(self, config: WaveletConfig) -> None:
        self.config = config
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for colormap LUT usage")
        self._cmap = _cm.get_cmap(self.config.cmap_name, 256)

    def to_chw(self, spec01: np.ndarray):
        spec01 = np.clip(spec01, 0.0, 1.0).astype(np.float32, copy=False)
        # Apply colormap: returns RGBA
        rgba = self._cmap(spec01)
        rgb = rgba[..., :3].astype(np.float32, copy=False)
        chw = np.transpose(rgb, (2, 0, 1))  # [3,H,W]
        if self.config.return_torch:
            if not _TORCH_AVAILABLE:
                raise ImportError("return_torch=True requires PyTorch to be installed")
            return torch.from_numpy(chw).contiguous()
        return chw


class CWTImagePipeline:
    """High-level orchestrator producing [3, output_size, output_size] from 1D signal."""

    def __init__(self, config: WaveletConfig) -> None:
        self.config = config
        self._scales = ScalesComputer(config)
        self._window = SignalWindowExtractor(config)
        self._cwt = CWTTransformer(config, self._scales)
        self._post = SpectrogramPostProcessor(config)
        self._export = RGBTensorExporter(config)

    def __call__(self, signal: np.ndarray | "torch.Tensor", start_idx: int = 0):
        window = self._window.extract(signal, start_idx=start_idx)
        coef, freqs_hz = self._cwt.compute(window)
        spec01 = self._post.process(coef, freqs_hz)
        rgb = self._export.to_chw(spec01)
        return rgb


# -----------------------------
# Convenience functional API
# -----------------------------


def build_config(**kwargs) -> WaveletConfig:
    return WaveletConfig(**kwargs)


def get_scales_and_freqs(config: WaveletConfig) -> Tuple[np.ndarray, np.ndarray]:
    sc = ScalesComputer(config)
    return sc.scales, sc.freqs_hz


def extract_window(signal: np.ndarray | "torch.Tensor", start_idx: int, config: WaveletConfig) -> np.ndarray:
    return SignalWindowExtractor(config).extract(signal, start_idx=start_idx)


def compute_cwt(window: np.ndarray, config: WaveletConfig) -> Tuple[np.ndarray, np.ndarray]:
    sc = ScalesComputer(config)
    return CWTTransformer(config, sc).compute(window)


def postprocess_spec(coef: np.ndarray, freqs_hz: np.ndarray, config: WaveletConfig) -> np.ndarray:
    return SpectrogramPostProcessor(config).process(coef, freqs_hz)


def to_rgb_chw(spec01: np.ndarray, config: WaveletConfig):
    return RGBTensorExporter(config).to_chw(spec01)


def make_cwt_image(signal: np.ndarray | "torch.Tensor", start_idx: int, config: WaveletConfig):
    return CWTImagePipeline(config)(signal, start_idx)


__all__ = [
    "WaveletConfig",
    "ScalesComputer",
    "SignalWindowExtractor",
    "CWTTransformer",
    "SpectrogramPostProcessor",
    "RGBTensorExporter",
    "CWTImagePipeline",
    "build_config",
    "get_scales_and_freqs",
    "extract_window",
    "compute_cwt",
    "postprocess_spec",
    "to_rgb_chw",
    "make_cwt_image",
]
