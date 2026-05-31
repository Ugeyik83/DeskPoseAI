"""
HRVAnalyzer: rPPG sinyalinden HRV tahmini.
Pan-Tompkins adaptasyonu — saf NumPy/SciPy, bağımlılık yok.
Deneysel modül — klinik kullanım için değil.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, medfilt
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
    MIN_SNR    = 4.0
    MIN_FRAMES = 200

    def __init__(self, buffer_size: int = 300):
        self._green_buffer: deque = deque(maxlen=buffer_size)
        self._time_buffer:  deque = deque(maxlen=buffer_size)

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        self._green_buffer.append(float(rgb_mean[1]))
        self._time_buffer.append(timestamp)

    # ── Pan-Tompkins adaptasyonu ──────────────────────────────────────────────
    @staticmethod
    def _pantompkins_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
        """Pan-Tompkins PPG adaptasyonu — türev + kare + MWI."""
        if len(signal) < int(fs * 1.0):
            return np.array([], dtype=int)

        # 1. Türev
        dy = np.diff(signal, prepend=signal[0])

        # 2. Kare
        dy2 = dy ** 2

        # 3. Moving window integration — 150ms pencere
        win = max(1, int(0.15 * fs))
        mwi = np.convolve(dy2, np.ones(win) / win, mode='same')

        # 4. Adaptive threshold
        threshold = np.mean(mwi) + 0.5 * np.std(mwi)

        # 5. Peak detection — 500ms minimum aralık
        min_delay = int(0.5 * fs)
        peaks = []
        last  = -min_delay

        above  = mwi > threshold
        diff   = np.diff(above.astype(int), prepend=0)
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        if len(ends) == 0 or len(starts) == 0:
            return np.array([], dtype=int)

        if ends[0] < starts[0]:
            ends = ends[1:]
        min_len = min(len(starts), len(ends))
        starts, ends = starts[:min_len], ends[:min_len]

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
        if len(self._green_buffer) < self.MIN_FRAMES:
            return None

        signal = np.array(self._green_buffer)
        times  = np.array(self._time_buffer)

        # 1. SNR kontrolü
        snr = float(np.var(signal) / (np.var(np.diff(signal)) + 1e-6))
        if snr < self.MIN_SNR:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Düşük SNR: {snr:.2f}")

        # 2. Cubic spline interpolasyon → TARGET_FS
        duration = times[-1] - times[0]
        if duration < 10.0:
            return None

        try:
            t_uniform     = np.arange(times[0], times[-1], 1.0 / self.TARGET_FS)
            interp_fn     = interp1d(times, signal, kind='cubic',
                                      bounds_error=False, fill_value='extrapolate')
            sig_resampled = interp_fn(t_uniform)
        except Exception:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="İnterpolasyon hatası")

        # 3. Detrend
        t_idx = np.arange(len(sig_resampled))
        sig_resampled = sig_resampled - np.polyval(
            np.polyfit(t_idx, sig_resampled, 1), t_idx)

        # 4. Medyan filtresi — küçük gürültü temizleme
        try:
            sig_resampled = medfilt(sig_resampled, kernel_size=5)
        except Exception:
            pass

        # 5. Bandpass filtre — 0.8–3Hz
        try:
            b, a = butter(3, [0.8, 3.0], btype='band', fs=self.TARGET_FS)
            sig_resampled = filtfilt(b, a, sig_resampled)
        except Exception:
            pass

        # 6. Normalize
        std = sig_resampled.std()
        if std < 1e-6:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="Düz sinyal")
        sig_resampled = (sig_resampled - np.mean(sig_resampled)) / std

        # 7. Pan-Tompkins peak detection
        peaks = self._pantompkins_peaks(sig_resampled, self.TARGET_FS)

        if len(peaks) < self.MIN_PEAKS:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Yetersiz tepe: {len(peaks)}")

        # 8. RR intervalları (ms)
        rr = np.diff(peaks) / self.TARGET_FS * 1000.0

        # Fizyolojik sınır filtresi
        rr = rr[(rr >= 400) & (rr <= 1200)]
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

        # 9. HRV metrikleri
        diff_rr = np.diff(rr)
        rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
        nn50    = int(np.sum(np.abs(diff_rr) > 50))
        pnn50   = float(nn50 / len(diff_rr) * 100)

        # Fizyolojik sınır kontrolü
        if not (8.0 <= rmssd <= 120.0):
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
        self._green_buffer.clear()
        self._time_buffer.clear()