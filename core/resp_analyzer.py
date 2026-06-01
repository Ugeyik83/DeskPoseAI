"""
RespAnalyzer: rPPG sinyalinden solunum hızı tahmini.
CHROM düşük bant filtresi + tepe tespiti.
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
    rate:     float   # nefes/dk
    snr:      float   # sinyal kalitesi
    reliable: bool
    reason:   str


class RespAnalyzer:
    TARGET_FS  = 10.0    # solunum için 10Hz yeterli
    MIN_FRAMES = 150     # ~15 sn @10fps eşdeğer
    RESP_LOW   = 0.1     # Hz — 6 nefes/dk
    RESP_HIGH  = 0.6     # Hz — 36 nefes/dk
    MIN_SNR    = 1.5

    def __init__(self, buffer_size: int = 600):
        # 600 frame @25fps = 24 sn
        self._rgb_buffer:  deque = deque(maxlen=buffer_size)
        self._time_buffer: deque = deque(maxlen=buffer_size)

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        """Her frame'de çağrılır. rgb_mean: [R, G, B]"""
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

    # ── Ana hesaplama ─────────────────────────────────────────────────────────
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
                              reason=f"Düşük SNR: {snr:.2f}")

        # 3. Cubic spline interpolasyon → TARGET_FS (10Hz)
        try:
            t_uniform     = np.linspace(times[0], times[-1],
                                         int(duration * self.TARGET_FS))
            cs            = CubicSpline(times, S)
            sig_resampled = cs(t_uniform)
        except Exception:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="İnterpolasyon hatası")

        # 4. Solunum bandpass — 0.1–0.6 Hz
        try:
            b, a = butter(3, [self.RESP_LOW, self.RESP_HIGH],
                          btype='band', fs=self.TARGET_FS)
            resp_signal = filtfilt(b, a, sig_resampled)
        except Exception:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="Filtre hatası")

        # 5. Normalize
        std = resp_signal.std()
        if std < 1e-6:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="Düz sinyal")
        resp_signal = (resp_signal - resp_signal.mean()) / std

        # 6. Tepe tespiti — minimum 1.5 sn aralık (max 40 nefes/dk)
        min_distance = int(self.TARGET_FS * 1.5)
        peaks, _ = find_peaks(resp_signal, distance=min_distance,
                               prominence=0.3)

        if len(peaks) < 3:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason=f"Yetersiz tepe: {len(peaks)}")

        # 7. Solunum hızı — tepe aralıklarından hesapla
        peak_times    = t_uniform[peaks]
        intervals     = np.diff(peak_times)  # sn cinsinden

        # Fizyolojik filtre — 1.5–10 sn arası (6–40 nefes/dk)
        intervals = intervals[(intervals >= 1.5) & (intervals <= 10.0)]

        if len(intervals) < 2:
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason="Geçerli interval yok")

        # Ortalama interval → nefes/dk
        avg_interval = float(np.median(intervals))
        rate         = round(60.0 / avg_interval, 1)

        # Fizyolojik sınır kontrolü
        if not (6.0 <= rate <= 40.0):
            return RespResult(rate=-1, snr=round(snr, 2), reliable=False,
                              reason=f"Sınır dışı: {rate:.1f}/dk")

        return RespResult(
            rate     = rate,
            snr      = round(snr, 2),
            reliable = True,
            reason   = "OK",
        )

    def reset(self):
        self._rgb_buffer.clear()
        self._time_buffer.clear()