"""
HRVAnalyzer: rPPG sinyalinden HRV tahmini.
CHROM + CIELAB a* füzyonu (saf NumPy) + Elgendi peak detection.
Deneysel modül — klinik kullanım için değil.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class HRVResult:
    rmssd:    float
    nn50:     int
    pnn50:    float
    snr:      float
    reliable: bool
    reason:   str


class HRVAnalyzer:
    TARGET_FS  = 100.0
    MIN_PEAKS  = 10
    MIN_SNR    = 2.0
    MIN_FRAMES = 200

    def __init__(self, buffer_size: int = 300):
        self._rgb_buffer:  deque = deque(maxlen=buffer_size)
        self._time_buffer: deque = deque(maxlen=buffer_size)

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        """Her frame'de çağrılır. rgb_mean: [R, G, B]"""
        self._rgb_buffer.append(rgb_mean.astype(np.float64))
        self._time_buffer.append(timestamp)

    # ── CIELAB a* dönüşümü — saf NumPy ──────────────────────────────────────
    @staticmethod
    def _rgb_to_astar(rgb_data: np.ndarray) -> np.ndarray:
        """RGB → CIELAB a* kanalı. Hareket artefaktına daha dirençli."""
        rgb_norm = np.clip(rgb_data / 255.0, 0, 1)

        # sRGB → linear RGB
        mask   = rgb_norm > 0.04045
        linear = np.where(mask,
                          ((rgb_norm + 0.055) / 1.055) ** 2.4,
                          rgb_norm / 12.92)

        # linear RGB → XYZ (D65)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz = linear @ M.T

        # XYZ → LAB
        xyz_n = np.array([0.95047, 1.00000, 1.08883])
        xyz_r = xyz / xyz_n

        epsilon = 0.008856
        kappa   = 903.3
        f = np.where(xyz_r > epsilon,
                     xyz_r ** (1.0 / 3.0),
                     (kappa * xyz_r + 16.0) / 116.0)

        # a* = 500 * (f_x - f_y)
        a_star = 500.0 * (f[:, 0] - f[:, 1])
        return a_star

    # ── CHROM sinyali ────────────────────────────────────────────────────────
    @staticmethod
    def _chrom_signal(rgb_data: np.ndarray) -> np.ndarray:
        """CHROM algoritması (De Haan & Jeanne 2013)."""
        R, G, B = rgb_data[:, 0], rgb_data[:, 1], rgb_data[:, 2]

        Rn = R / (R.mean() + 1e-6)
        Gn = G / (G.mean() + 1e-6)
        Bn = B / (B.mean() + 1e-6)

        Xs = 3 * Rn - 2 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn
        S  = Xs - (Xs.std() / (Ys.std() + 1e-6)) * Ys

        t_idx = np.arange(len(S))
        S = S - np.polyval(np.polyfit(t_idx, S, 1), t_idx)

        return S

    # ── Elgendi (2013) peak detection ────────────────────────────────────────
    @staticmethod
    def _elgendi_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
        if len(signal) < int(fs * 1.0):
            return np.array([], dtype=int)

        sqrd = signal ** 2

        peak_w = max(1, int(np.round(0.111 * fs)))
        beat_w = max(1, int(np.round(0.667 * fs)))

        ma_peak = uniform_filter1d(sqrd, size=peak_w)
        ma_beat = uniform_filter1d(sqrd, size=beat_w)

        offset = 0.05 * np.mean(sqrd)
        thresh = ma_beat + offset

        blocks = (ma_peak > thresh).astype(int)
        diff   = np.diff(blocks, prepend=0)
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        if len(ends) == 0 or len(starts) == 0:
            return np.array([], dtype=int)

        if ends[0] < starts[0]:
            ends = ends[1:]
        min_len = min(len(starts), len(ends))
        starts, ends = starts[:min_len], ends[:min_len]

        min_delay = int(0.45 * fs)
        peaks = []
        last  = -min_delay

        for s, e in zip(starts, ends):
            if e <= s:
                continue
            peak = s + int(np.argmax(signal[s:e]))
            if peak - last >= min_delay:
                peaks.append(peak)
                last = peak

        return np.array(peaks, dtype=int)

    # ── Ana hesaplama ─────────────────────────────────────────────────────────
    def compute(self) -> Optional[HRVResult]:
        if len(self._rgb_buffer) < self.MIN_FRAMES:
            return None

        rgb_data = np.array(self._rgb_buffer)
        times    = np.array(self._time_buffer)

        # 1. CHROM sinyali
        try:
            S = self._chrom_signal(rgb_data)
        except Exception as e:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                            snr=-1, reliable=False,
                            reason=f"CHROM hatası: {str(e)[:20]}")

        # 2. SNR kontrolü
        snr = float(np.var(S) / (np.var(np.diff(S)) + 1e-6))
        if snr < self.MIN_SNR:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Düşük SNR: {snr:.2f}")

        # 3. Cubic spline interpolasyon → TARGET_FS
        duration = times[-1] - times[0]
        if duration < 10.0:
            return None

        try:
            t_uniform     = np.arange(times[0], times[-1], 1.0 / self.TARGET_FS)
            interp_fn     = interp1d(times, S, kind='cubic',
                                      bounds_error=False, fill_value='extrapolate')
            sig_resampled = interp_fn(t_uniform)
        except Exception:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="İnterpolasyon hatası")

        # 4. Bandpass filtre — 0.8–3Hz
        try:
            b, a = butter(3, [0.7, 2.5], btype='band', fs=self.TARGET_FS)
            sig_resampled = filtfilt(b, a, sig_resampled)
        except Exception:
            pass

        # 5. Normalize
        std = sig_resampled.std()
        if std < 1e-6:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="Düz sinyal")
        sig_resampled = (sig_resampled - np.mean(sig_resampled)) / std

        # 6. Elgendi peak detection
        peaks = self._elgendi_peaks(sig_resampled, self.TARGET_FS)

        if len(peaks) < self.MIN_PEAKS:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Yetersiz tepe: {len(peaks)}")

        # 7. RR intervalları (ms)
        rr = np.diff(peaks) / self.TARGET_FS * 1000.0

        # Fizyolojik sınır filtresi
        rr = rr[(rr >= 400) & (rr <= 1000)]
        if len(rr) < 4:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="Geçerli RR interval yok")

        # Adaptif outlier filtresi
        rr_median = np.median(rr)
        rr_std    = np.std(rr)
        rr        = rr[np.abs(rr - rr_median) < 2.5 * rr_std]

        # MAD filtresi
        rr_mad = np.median(np.abs(rr - np.median(rr)))
        rr     = rr[np.abs(rr - np.median(rr)) < 3 * rr_mad]

        if len(rr) < 4:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="Filtre sonrası yetersiz")

        # 8. HRV metrikleri
        diff_rr = np.diff(rr)
        rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
        nn50    = int(np.sum(np.abs(diff_rr) > 50))
        pnn50   = float(nn50 / len(diff_rr) * 100)

        # Fizyolojik sınır kontrolü
        if not (8.0 <= rmssd <= 400.0):
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Sınır dışı: {rmssd:.1f}ms")

        return HRVResult(
            rmssd    = round(rmssd, 1),
            nn50     = nn50,
            pnn50    = round(pnn50, 1),
            snr      = round(snr, 2),
            reliable = True,
            reason   = "OK",
        )

    def reset(self):
        self._rgb_buffer.clear()
        self._time_buffer.clear()