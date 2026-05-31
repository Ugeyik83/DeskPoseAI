"""
HRVAnalyzer: rPPG sinyalinden HRV tahmini.
NeuroKit2 tabanlı — ppg_clean + ppg_findpeaks + hrv_time pipeline.
Deneysel modül — klinik kullanım için değil.
"""

import numpy as np
from scipy.interpolate import interp1d
from collections import deque
from dataclasses import dataclass
from typing import Optional

try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False


@dataclass
class HRVResult:
    rmssd:    float
    nn50:     int
    pnn50:    float
    snr:      float
    reliable: bool
    reason:   str


class HRVAnalyzer:
    TARGET_FS  = 100.0   # interpolasyon hedef frekansı (Hz)
    MIN_PEAKS  = 10      # güvenilir sonuç için minimum tepe sayısı
    MIN_SNR    = 3.0     # minimum sinyal/gürültü oranı
    MIN_FRAMES = 200     # minimum buffer (~13 sn @15fps)

    def __init__(self, buffer_size: int = 450):
        self._green_buffer: deque = deque(maxlen=buffer_size)
        self._time_buffer:  deque = deque(maxlen=buffer_size)

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        """Her frame'de çağrılır. rgb_mean: [R, G, B]"""
        self._green_buffer.append(float(rgb_mean[1]))  # yeşil kanal
        self._time_buffer.append(timestamp)

    def compute(self) -> Optional[HRVResult]:
        if not NK_AVAILABLE:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=-1, reliable=False,
                             reason="neurokit2 yüklü değil")

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

        # 2. Cubic spline interpolasyon → TARGET_FS (100Hz)
        duration = times[-1] - times[0]
        if duration <= 1.0:
            return None

        t_uniform = np.arange(times[0], times[-1], 1.0 / self.TARGET_FS)
        try:
            interp_fn = interp1d(times, signal, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
            signal_resampled = interp_fn(t_uniform)
        except Exception:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="İnterpolasyon hatası")

        # 3. NeuroKit2 — PPG temizleme + peak detection
        try:
            ppg_clean = nk.ppg_clean(
                signal_resampled,
                sampling_rate=int(self.TARGET_FS)
            )
            peaks_info = nk.ppg_findpeaks(
                ppg_clean,
                sampling_rate=int(self.TARGET_FS),
                method="elgendi"
            )
            peaks = peaks_info["PPG_Peaks"]
        except Exception as e:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"NK2 peak hatası: {str(e)[:30]}")

        if len(peaks) < self.MIN_PEAKS:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Yetersiz tepe: {len(peaks)}")

        # 4. NeuroKit2 — HRV zaman alanı metrikleri
        try:
            hrv_df = nk.hrv_time(
                peaks_info,
                sampling_rate=int(self.TARGET_FS)
            )
            rmssd = float(hrv_df["HRV_RMSSD"].values[0])
            nn50  = int(hrv_df["HRV_pNN50"].values[0] * len(peaks) / 100)
            pnn50 = float(hrv_df["HRV_pNN50"].values[0])
        except Exception as e:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"HRV hesap hatası: {str(e)[:30]}")

        # 5. Fizyolojik sınır kontrolü
        if not (5 <= rmssd <= 150):
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason=f"Fizyolojik sınır dışı: {rmssd:.1f}ms")

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