"""
HRVAnalyzer: rPPG sinyalinden HRV tahmini.
Deneysel modül — klinik kullanım için değil.
Sinyal kalitesi düşükse otomatik devre dışı kalır.
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class HRVResult:
    rmssd:      float   # ms — ana HRV metriği
    nn50:       int     # 50ms üzeri ardışık fark sayısı
    pnn50:      float   # nn50 / toplam interval %
    snr:        float   # sinyal kalite skoru
    reliable:   bool    # güvenilir mi?
    reason:     str     # güvenilir değilse neden


class HRVAnalyzer:
    TARGET_FS   = 100.0   # interpolasyon hedef frekansı (Hz)
    BANDPASS_LO = 0.7     # Hz (~42 BPM)
    BANDPASS_HI = 4.0     # Hz (~240 BPM)
    MIN_PEAKS   = 8       # güvenilir sonuç için minimum tepe sayısı
    MIN_SNR     = 2.0     # minimum sinyal/gürültü oranı

    def __init__(self, buffer_size: int = 450):
        self._raw_buffer: deque = deque(maxlen=buffer_size)
        self._time_buffer: deque = deque(maxlen=buffer_size)

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        """Her frame'de çağrılır. rgb_mean: [R, G, B]"""
        self._raw_buffer.append(float(rgb_mean[1]))  # yeşil kanal
        self._time_buffer.append(timestamp)

    def compute(self) -> Optional[HRVResult]:
        if len(self._raw_buffer) < 150:   # en az 10 sn
            return None

        signal = np.array(self._raw_buffer)
        times  = np.array(self._time_buffer)

        # 1. Detrend
        signal = signal - np.polyval(np.polyfit(np.arange(len(signal)), signal, 1),
                                      np.arange(len(signal)))

        # 2. SNR kontrolü
        snr = float(np.var(signal) / (np.var(np.diff(signal)) + 1e-6))
        if snr < self.MIN_SNR:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=snr, reliable=False,
                             reason=f"Düşük SNR: {snr:.2f}")

        # 3. Interpolasyon: gerçek FPS → 100Hz
        duration = times[-1] - times[0]
        if duration <= 0:
            return None
        t_uniform = np.arange(times[0], times[-1], 1.0 / self.TARGET_FS)
        interp_fn = interp1d(times, signal, kind='cubic', bounds_error=False,
                              fill_value='extrapolate')
        signal_resampled = interp_fn(t_uniform)

        # 4. Bandpass filtre
        nyq = self.TARGET_FS / 2.0
        lo  = self.BANDPASS_LO / nyq
        hi  = self.BANDPASS_HI / nyq
        b, a = butter(3, [lo, hi], btype='band')
        try:
            signal_filtered = filtfilt(b, a, signal_resampled)
        except Exception:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=snr, reliable=False,
                             reason="Filtre hatası")

        # 5. Peak detection
        min_distance = int(self.TARGET_FS * 0.4)  # min 400ms arası (~150 BPM max)
        peaks, props = find_peaks(signal_filtered,
                                   distance=min_distance,
                                   prominence=np.std(signal_filtered) * 0.5)

        if len(peaks) < self.MIN_PEAKS:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=snr, reliable=False,
                             reason=f"Yetersiz tepe: {len(peaks)}")

        # 6. RR intervalları (ms)
        rr_intervals = np.diff(peaks) / self.TARGET_FS * 1000.0

        # Fizyolojik sınır filtresi (300–2000ms arası geçerli)
        rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]

        if len(rr_intervals) < 4:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=snr, reliable=False,
                             reason="Geçerli RR interval yok")

        # 7. HRV metrikleri
        diff_rr = np.diff(rr_intervals)
        rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
        nn50    = int(np.sum(np.abs(diff_rr) > 50))
        pnn50   = float(nn50 / len(diff_rr) * 100)

        return HRVResult(
            rmssd    = round(rmssd, 1),
            nn50     = nn50,
            pnn50    = round(pnn50, 1),
            snr      = round(snr, 2),
            reliable = True,
            reason   = "OK",
        )

    def reset(self):
        self._raw_buffer.clear()
        self._time_buffer.clear()