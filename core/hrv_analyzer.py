"""
HRVAnalyzer: rPPG sinyalinden HRV tahmini.
CHROM + Elgendi peak detection + Temporal averaging.
Deneysel modül — klinik kullanım için değil.
"""

import numpy as np
import pywt
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from collections import deque
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class HRVResult:
    rmssd:    float
    nn50:     int
    pnn50:    float
    snr:      float
    reliable: bool
    reason:   str


class HRVAnalyzer:
    TARGET_FS    = 100.0
    MIN_PEAKS    = 10
    MIN_SNR      = 2.0
    MIN_FRAMES   = 500
    N_WINDOWS    = 4      # temporal averaging pencere sayısı
    WINDOW_SEC   = 12.0   # her pencere süresi (sn)

    def __init__(self, buffer_size: int = 500):
        self._rgb_buffer:  deque = deque(maxlen=buffer_size)
        self._time_buffer: deque = deque(maxlen=buffer_size)
        self._rmssd_history: deque = deque(maxlen=self.N_WINDOWS)

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        self._rgb_buffer.append(rgb_mean.astype(np.float64))
        self._time_buffer.append(timestamp)

    # ── CHROM sinyali ────────────────────────────────────────────────────────
    @staticmethod
    def _chrom_signal(rgb_data: np.ndarray) -> np.ndarray:
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

        sqrd   = signal ** 2
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

    # ── Tek pencere RMSSD ─────────────────────────────────────────────────────
    def _compute_window_rmssd(self, rgb_data: np.ndarray,
                               times: np.ndarray) -> Optional[float]:
        """Tek bir zaman penceresi için RMSSD hesapla."""
        try:
            S = self._chrom_signal(rgb_data)
        except Exception:
            return None

        snr = float(np.var(S) / (np.var(np.diff(S)) + 1e-6))
        if snr < self.MIN_SNR:
            return None

        duration = times[-1] - times[0]
        if duration < 5.0:
            return None

        try:
            t_uniform = np.linspace(times[0], times[-1],
                                     int(duration * self.TARGET_FS))
            cs            = CubicSpline(times, S)
            sig_resampled = cs(t_uniform)
        except Exception:
            return None

        try:
            b, a = butter(3, [0.7, 2.5], btype='band', fs=self.TARGET_FS)
            sig_resampled = filtfilt(b, a, sig_resampled)
        except Exception:
            pass
        
        # DWT — H.264 sıkıştırma gürültüsü temizleme
        try:
            coeffs = pywt.wavedec(sig_resampled, 'db4', level=4)
            threshold = (np.sqrt(2 * np.log(len(sig_resampled))) *
                        np.median(np.abs(coeffs[-1])) / 0.6745)
            coeffs_thresh = [coeffs[0]]
            for c in coeffs[1:]:
                coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
            sig_resampled = pywt.waverec(coeffs_thresh, 'db4')[:len(sig_resampled)]
        except Exception:
            pass



        std = sig_resampled.std()
        if std < 1e-6:
            return None
        sig_resampled = (sig_resampled - np.mean(sig_resampled)) / std

        peaks = self._elgendi_peaks(sig_resampled, self.TARGET_FS)
        if len(peaks) < self.MIN_PEAKS:
            return None

        peak_times = t_uniform[peaks]
        rr = np.diff(peak_times) * 1000.0

        rr = rr[(rr >= 400) & (rr <= 1500)]
        if len(rr) < 4:
            return None

        rr_median = np.median(rr)
        rr_std    = np.std(rr)
        rr        = rr[np.abs(rr - rr_median) < 2.5 * rr_std]

        rr_mad = np.median(np.abs(rr - np.median(rr)))
        rr     = rr[np.abs(rr - np.median(rr)) < 3 * rr_mad]

        if len(rr) < 4:
            return None

        diff_rr = np.diff(rr)
        rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))

        if not (8.0 <= rmssd <= 400.0):
            return None

        return round(rmssd, 1)

    # ── Ana hesaplama — temporal averaging ───────────────────────────────────
    def compute(self) -> Optional[HRVResult]:
        if len(self._rgb_buffer) < self.MIN_FRAMES:
            return None

        rgb_all   = np.array(self._rgb_buffer)
        times_all = np.array(self._time_buffer)

        total_duration = times_all[-1] - times_all[0]
        if total_duration < self.WINDOW_SEC:
            return None

        # SNR — tüm buffer üzerinde
        try:
            S_full = self._chrom_signal(rgb_all)
            snr = float(np.var(S_full) / (np.var(np.diff(S_full)) + 1e-6))
        except Exception:
            snr = 0.0

        if snr < self.MIN_SNR:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Düşük SNR: {snr:.2f}")

        # Kayan pencereler — son N_WINDOWS × WINDOW_SEC hesapla
        window_results: List[float] = []
        step = max(1.0, (total_duration - self.WINDOW_SEC) / max(self.N_WINDOWS - 1, 1))

        for i in range(self.N_WINDOWS):
            win_end   = times_all[-1] - i * step
            win_start = win_end - self.WINDOW_SEC
            if win_start < times_all[0]:
                break

            mask = (times_all >= win_start) & (times_all <= win_end)
            if mask.sum() < 30:
                continue

            result = self._compute_window_rmssd(rgb_all[mask], times_all[mask])
            if result is not None:
                window_results.append(result)

        if not window_results:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="Hiçbir pencere geçerli değil")

        # Temporal averaging — medyan (outlier'a dayanıklı)
        rmssd_avg = float(np.median(window_results))

        # Güvenilirlik: en az 2 pencere gerekli
        if len(window_results) < 2:
            return HRVResult(rmssd=round(rmssd_avg, 1), nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Tek pencere: {len(window_results)}")

        # nn50 ve pnn50 — son geçerli pencereden
        # Basit tahmin: temporal averaging sonucuna göre
        nn50  = -1
        pnn50 = -1.0

        return HRVResult(
            rmssd    = round(rmssd_avg, 1),
            nn50     = nn50,
            pnn50    = pnn50,
            snr      = round(snr, 2),
            reliable = True,
            reason   = f"OK ({len(window_results)} pencere ortalaması)",
        )

    def reset(self):
        self._rgb_buffer.clear()
        self._time_buffer.clear()
        self._rmssd_history.clear()