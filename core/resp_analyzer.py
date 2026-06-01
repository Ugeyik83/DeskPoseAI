"""
RespAnalyzer: rPPG sinyalinden solunum hızı tahmini.
CHROM düşük bant filtresi + tepe tespiti + kümülatif sayaç.
Deneysel modül — klinik kullanım için değil.
Normal solunum: 12–20 nefes/dk (0.2–0.33 Hz)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class RespResult:
    rate:          float
    snr:           float
    reliable:      bool
    reason:        str
    breath_count:  int = 0


class RespAnalyzer:
    TARGET_FS  = 10.0
    MIN_FRAMES = 150
    RESP_LOW   = 0.1
    RESP_HIGH  = 0.6
    MIN_SNR    = 1.5

    def __init__(self, buffer_size: int = 600):
        self._rgb_buffer:      deque = deque(maxlen=buffer_size)
        self._time_buffer:     deque = deque(maxlen=buffer_size)
        self._total_breaths:   int   = 0
        self._last_peak_time:  float = 0.0

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        self._rgb_buffer.append(rgb_mean.astype(np.float64))
        self._time_buffer.append(timestamp)

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

    def compute(self) -> Optional[RespResult]:
        if len(self._rgb_buffer) < self.MIN_FRAMES:
            return None

        rgb_data = np.array(self._rgb_buffer)
        times    = np.array(self._time_buffer)

        duration = times[-1] - times[0]
        if duration < 15.0:
            return None

        # 1. CHROM sinyali
        try:
            S = self._chrom_signal(rgb_data)
        except Exception as e:
            return RespResult(rate=-1, snr=-1, reliable=False,
                              reason=f"CHROM hatası: {str(e)[:20]}")

        # 2. SNR kontrolü
        snr = float(np.var(S) / (np.var(np.diff(S)) + 1e-6))
        if snr < self.MIN_SNR:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason=f"Düşük SNR: {snr:.2f}",
                              breath_count=self._total_breaths)

        # 3. Cubic spline interpolasyon → TARGET_FS
        try:
            t_uniform     = np.linspace(times[0], times[-1],
                                         int(duration * self.TARGET_FS))
            cs            = CubicSpline(times, S)
            sig_resampled = cs(t_uniform)
        except Exception:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="İnterpolasyon hatası",
                              breath_count=self._total_breaths)

        # 4. Solunum bandpass — 0.1–0.6 Hz
        try:
            b, a = butter(3, [self.RESP_LOW, self.RESP_HIGH],
                          btype='band', fs=self.TARGET_FS)
            resp_signal = filtfilt(b, a, sig_resampled)
        except Exception:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="Filtre hatası",
                              breath_count=self._total_breaths)

        # 5. Normalize
        std = resp_signal.std()
        if std < 1e-6:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="Düz sinyal",
                              breath_count=self._total_breaths)
        resp_signal = (resp_signal - resp_signal.mean()) / std

        # 6. Tepe tespiti
        min_distance = int(self.TARGET_FS * 1.5)
        peaks, _ = find_peaks(resp_signal, distance=min_distance,
                               prominence=0.3)

        if len(peaks) < 3:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason=f"Yetersiz tepe: {len(peaks)}",
                              breath_count=self._total_breaths)

        # 7. Kümülatif nefes sayacı
        if len(peaks) >= 1:
            last_peak_t = float(t_uniform[peaks[-1]])
            if last_peak_t > self._last_peak_time:
                new_peaks = sum(1 for p in t_uniform[peaks]
                                if p > self._last_peak_time)
                self._total_breaths += new_peaks
                self._last_peak_time  = last_peak_t

        # 8. Solunum hızı
        peak_times = t_uniform[peaks]
        intervals  = np.diff(peak_times)
        intervals  = intervals[(intervals >= 1.5) & (intervals <= 10.0)]

        if len(intervals) < 2:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="Geçerli interval yok",
                              breath_count=self._total_breaths)

        avg_interval = float(np.median(intervals))
        rate         = round(60.0 / avg_interval, 1)

        if not (6.0 <= rate <= 40.0):
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason=f"Sınır dışı: {rate:.1f}/dk",
                              breath_count=self._total_breaths)

        return RespResult(
            rate          = rate,
            snr           = round(snr, 2),
            reliable      = True,
            reason        = "OK",
            breath_count  = self._total_breaths,
        )

    def reset(self):
        self._rgb_buffer.clear()
        self._time_buffer.clear()
        self._total_breaths  = 0
        self._last_peak_time = 0.0