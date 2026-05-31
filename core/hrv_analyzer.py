"""
HRVAnalyzer: rPPG sinyalinden HRV tahmini.
Elgendi (2013) peak detection — saf NumPy/SciPy, bağımlılık yok.
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
    TARGET_FS  = 100.0   # interpolasyon hedef frekansı (Hz)
    MIN_PEAKS  = 10      # güvenilir sonuç için minimum tepe sayısı
    MIN_SNR    = 4.0     # minimum sinyal/gürültü oranı
    MIN_FRAMES = 200     # minimum buffer (~13 sn @15fps)

    def __init__(self, buffer_size: int = 450):
        self._green_buffer: deque = deque(maxlen=buffer_size)
        self._time_buffer:  deque = deque(maxlen=buffer_size)

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        """Her frame'de çağrılır. rgb_mean: [R, G, B]"""
        self._green_buffer.append(float(rgb_mean[1]))  # yeşil kanal
        self._time_buffer.append(timestamp)

    # ── Elgendi (2013) peak detection ────────────────────────────────────────
    @staticmethod
    def _elgendi_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Elgendi M. et al. (2013) PLoS ONE — systolic peak detection.
        peakwindow: 0.111 sn, beatwindow: 0.667 sn, offset: 0.02, mindelay: 0.3 sn
        """
        if len(signal) < int(fs * 1.0):
            return np.array([], dtype=int)

        # Mutlak değer — rPPG için kare almaktan daha uygun
        sqrd = np.abs(signal)

        # Peak penceresi ve beat penceresi
        peak_w = max(1, int(np.round(0.111 * fs)))
        beat_w = max(1, int(np.round(0.667 * fs)))

        ma_peak = uniform_filter1d(sqrd, size=peak_w)
        ma_beat = uniform_filter1d(sqrd, size=beat_w)

        # Offset eşiği
        offset  = 0.02 * np.mean(sqrd)
        thresh  = ma_beat + offset

        # Blok tespiti
        blocks  = (ma_peak > thresh).astype(int)
        diff    = np.diff(blocks, prepend=0)
        starts  = np.where(diff == 1)[0]
        ends    = np.where(diff == -1)[0]

        if len(ends) == 0 or len(starts) == 0:
            return np.array([], dtype=int)

        # Uzunluk eşleştir
        if ends[0] < starts[0]:
            ends = ends[1:]
        min_len = min(len(starts), len(ends))
        starts, ends = starts[:min_len], ends[:min_len]

        # Her blokta maksimum noktayı bul
        min_delay = int(0.4 * fs)
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

        # 3.5 Wavelet alt bant seçimi — WaveHRV yaklaşımı

        try:
            from scipy.signal import cwt, morlet2
            widths = np.arange(1, 40)
            cwtmatr = cwt(sig_resampled, morlet2, widths)
            lo_idx = max(0, int(self.TARGET_FS / (3.0 * 2 * np.pi)))
            hi_idx = min(len(widths), int(self.TARGET_FS / (0.7 * 2 * np.pi)))
            cardiac_rows = cwtmatr[lo_idx:hi_idx, :]
            if cardiac_rows.shape[0] > 0:
                # np.abs yok — gerçek sinyal değerlerini al
                sig_resampled = np.mean(cardiac_rows, axis=0)
        except Exception:
            pass

        # 4. Bandpass filtre — 0.7–4Hz hareket artefaktı bastırma
        try:
            b, a = butter(3, [0.7, 4.0], btype='band', fs=self.TARGET_FS)
            sig_resampled = filtfilt(b, a, sig_resampled)
        except Exception:
            pass  # filtre başarısız olursa devam et

        # 5. Normalize — ortalama çıkar, std'ye böl
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
        rr = rr[(rr >= 400) & (rr <= 1200)]
        if len(rr) < 4:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="Geçerli RR interval yok")

        # MAD outlier filtresi
        rr_median = np.median(rr)
        rr_mad    = np.median(np.abs(rr - rr_median))
        rr        = rr[np.abs(rr - rr_median) < 3 * rr_mad]
        if len(rr) < 4:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), reliable=False,
                             reason="MAD filtresi sonrası yetersiz")

        # 8. HRV metrikleri
        diff_rr = np.diff(rr)
        rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
        nn50    = int(np.sum(np.abs(diff_rr) > 50))
        pnn50   = float(nn50 / len(diff_rr) * 100)

        # Fizyolojik sınır kontrolü
        if not (5.0 <= rmssd <= 250.0):
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