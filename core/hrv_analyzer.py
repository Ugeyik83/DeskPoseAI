"""
HRVAnalyzer: rPPG sinyalinden HRV tahmini.
CHROM + Ten rengi kalibrasyonu + Elgendi peak detection + Temporal averaging.
Deneysel modül — klinik kullanım için değil.
"""

import numpy as np
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
    bpm:      float
    rr_std:   float
    reliable: bool
    reason:   str


class HRVAnalyzer:
    TARGET_FS        = 100.0
    MIN_PEAKS        = 20
    MIN_SNR          = 2.0
    MIN_FRAMES       = 500
    MIN_DURATION_SEC = 30.0   # FPS bağımsız süre kontrolü
    N_WINDOWS        = 3
    WINDOW_SEC       = 30.0
    MIN_HR_BPM       = 45
    MAX_HR_BPM       = 160

    def __init__(self, buffer_size: int = 1500):
        self._rgb_buffer:     deque      = deque(maxlen=buffer_size)
        self._time_buffer:    deque      = deque(maxlen=buffer_size)
        self._rmssd_history:  deque      = deque(maxlen=self.N_WINDOWS)
        self._chrom_gain:     np.ndarray = None
        self._gain_computed:  bool       = False

    def add_sample(self, rgb_mean: np.ndarray, timestamp: float):
        self._rgb_buffer.append(rgb_mean.astype(np.float64))
        self._time_buffer.append(timestamp)

    # ── Ten rengi kazanç kalibrasyonu ────────────────────────────────────────
    @staticmethod
    def _compute_gain(rgb_data: np.ndarray) -> np.ndarray:
        R_mean = rgb_data[:, 0].mean()
        G_mean = rgb_data[:, 1].mean()
        B_mean = rgb_data[:, 2].mean()
        total   = R_mean + G_mean + B_mean + 1e-6
        r_ratio = R_mean / total
        g_ratio = G_mean / total
        r_gain  = 1.0 + max(0.0, 0.33 - g_ratio) * 2.0
        g_gain  = 1.0
        b_gain  = 1.0 + max(0.0, 0.33 - r_ratio) * 0.5
        return np.array([r_gain, g_gain, b_gain])

    # ── CHROM sinyali ────────────────────────────────────────────────────────
    @staticmethod
    def _chrom_signal(rgb_data: np.ndarray,
                      gain: np.ndarray = None) -> np.ndarray:
        R, G, B = rgb_data[:, 0], rgb_data[:, 1], rgb_data[:, 2]
        if gain is not None:
            R = R * gain[0]
            G = G * gain[1]
            B = B * gain[2]
        Rn = R / (R.mean() + 1e-6)
        Gn = G / (G.mean() + 1e-6)
        Bn = B / (B.mean() + 1e-6)
        Xs = 3 * Rn - 2 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn
        S  = Xs - (Xs.std() / (Ys.std() + 1e-6)) * Ys
        t_idx = np.arange(len(S))
        S = S - np.polyval(np.polyfit(t_idx, S, 1), t_idx)
        return S

    # ── Elgendi (2013) peak detection — rectified ────────────────────────────
    @staticmethod
    def _elgendi_peaks(signal: np.ndarray, fs: float,
                       max_hr_bpm: float = 160) -> np.ndarray:
        if len(signal) < int(fs * 1.0):
            return np.array([], dtype=int)

        # Rectify — negatif değerleri sıfırla, ters polarite etkisini önle
        rectified = signal.copy()
        rectified[rectified < 0] = 0
        sqrd = rectified ** 2

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

        # HR üst sınırına göre min_delay
        min_delay = int((60.0 / max_hr_bpm) * fs)
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
                               times: np.ndarray) -> Optional[dict]:
        # Timestamp sıralama + unique kontrolü
        order = np.argsort(times)
        times = times[order]
        rgb_data = rgb_data[order]
        unique_mask = np.diff(times, prepend=times[0] - 1e-6) > 0
        times    = times[unique_mask]
        rgb_data = rgb_data[unique_mask]
        if len(times) < 4:
            return None

        try:
            S = self._chrom_signal(rgb_data, gain=self._chrom_gain)
        except Exception:
            return None

        snr = float(np.var(S) / (np.var(np.diff(S)) + 1e-6))
        if snr < self.MIN_SNR:
            return None

        duration = times[-1] - times[0]
        if duration < 5.0:
            return None

        try:
            t_uniform     = np.arange(times[0], times[-1], 1.0 / self.TARGET_FS)
            cs            = CubicSpline(times, S)
            sig_resampled = cs(t_uniform)
        except Exception:
            return None

        try:
            b, a = butter(3, [0.7, 2.5], btype='band', fs=self.TARGET_FS)
            sig_resampled = filtfilt(b, a, sig_resampled)
        except Exception:
            return None

        std = sig_resampled.std()
        if std < 1e-6:
            return None
        sig_resampled = (sig_resampled - np.mean(sig_resampled)) / std

        peaks = self._elgendi_peaks(sig_resampled, self.TARGET_FS,
                                     self.MAX_HR_BPM)
        if len(peaks) < self.MIN_PEAKS:
            return None

        peak_times = t_uniform[peaks]
        rr = np.diff(peak_times) * 1000.0

        # Fizyolojik sınır
        rr = rr[(rr >= 400) & (rr <= 1500)]
        if len(rr) < 4:
            return None

        # HR BPM kontrolü
        mean_rr = np.mean(rr)
        hr_bpm  = 60000.0 / mean_rr
        if not (self.MIN_HR_BPM <= hr_bpm <= self.MAX_HR_BPM):
            return None

        # RR std kontrolü — çok saçma dağılımı reddet
        rr_std_raw = float(np.std(rr))
        if rr_std_raw > 250:
            return None

        # Adaptif outlier filtresi
        rr_median = np.median(rr)
        rr_std    = float(np.std(rr))
        if rr_std > 1e-6:
            rr = rr[np.abs(rr - rr_median) < 2.5 * rr_std]

        # MAD filtresi — rr_mad == 0 koruması
        rr_mad = np.median(np.abs(rr - np.median(rr)))
        if rr_mad > 1e-6:
            rr = rr[np.abs(rr - np.median(rr)) < 3 * rr_mad]

        if len(rr) < 4:
            return None

        # %20 ardışık fark filtresi
        diff_rr        = np.diff(rr)
        threshold      = 0.20 * rr[:-1]
        valid_mask     = np.abs(diff_rr) < threshold
        filtered_diffs = diff_rr[valid_mask]

        if len(filtered_diffs) == 0:
            return None

        rmssd = float(np.sqrt(np.mean(filtered_diffs ** 2)))

        if not (8.0 <= rmssd <= 180.0):
            return None

        return {
            "rmssd": round(rmssd, 1),
            "bpm": round(float(hr_bpm), 1),
            "rr_std": round(float(np.std(rr)), 1),
        }

    # ── Ana hesaplama — temporal averaging ───────────────────────────────────
    def compute(self) -> Optional[HRVResult]:
        if len(self._rgb_buffer) < self.MIN_FRAMES:
            return None

        rgb_all   = np.array(self._rgb_buffer)
        times_all = np.array(self._time_buffer)

        # Timestamp sıralama + unique kontrolü
        order = np.argsort(times_all)
        times_all = times_all[order]
        rgb_all   = rgb_all[order]
        unique_mask = np.diff(times_all, prepend=times_all[0] - 1e-6) > 0
        times_all = times_all[unique_mask]
        rgb_all   = rgb_all[unique_mask]

        total_duration = times_all[-1] - times_all[0]
        if total_duration < self.MIN_DURATION_SEC:
            return None

        # Gain hesaplama — compute() başında, SNR'dan önce
        if not self._gain_computed:
            self._chrom_gain    = self._compute_gain(rgb_all)
            self._gain_computed = True
        else:
            # Adaptif gain — yavaş güncelleme
            new_gain = self._compute_gain(rgb_all)
            self._chrom_gain = 0.9 * self._chrom_gain + 0.1 * new_gain

        # SNR kontrolü — gain uygulanmış sinyal üzerinde
        try:
            S_full = self._chrom_signal(rgb_all, gain=self._chrom_gain)
            snr    = float(np.var(S_full) / (np.var(np.diff(S_full)) + 1e-6))
        except Exception:
            snr = 0.0

        if snr < self.MIN_SNR:
            return HRVResult(rmssd=-1, nn50=-1, pnn50=-1,
                             snr=round(snr, 2), bpm=-1, rr_std=-1,
                             reliable=False,
                             reason=f"Düşük SNR: {snr:.2f}")

        # Kayan pencereler
        window_results: List[dict] = []
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
                             snr=round(snr, 2), bpm=-1, rr_std=-1,
                             reliable=False,
                             reason="Hiçbir pencere geçerli değil")

        # Temporal averaging — medyan + history smoothing
        rmssd_avg = float(np.median([r["rmssd"] for r in window_results]))
        bpm_avg = float(np.median([r["bpm"] for r in window_results]))
        rr_std_avg = float(np.median([r["rr_std"] for r in window_results]))
        self._rmssd_history.append(rmssd_avg)
        rmssd_smooth = float(np.median(self._rmssd_history))

        if len(window_results) < 2:
            return HRVResult(rmssd=round(rmssd_smooth, 1), nn50=-1, pnn50=-1,
                             snr=round(snr, 2), bpm=round(bpm_avg, 1),
                             rr_std=round(rr_std_avg, 1), reliable=False,
                             reason=f"Tek pencere: {len(window_results)}")

        return HRVResult(
            rmssd    = round(rmssd_smooth, 1),
            nn50     = -1,
            pnn50    = -1.0,
            snr      = round(snr, 2),
            bpm      = round(bpm_avg, 1),
            rr_std   = round(rr_std_avg, 1),
            reliable = True,
            reason   = f"OK ({len(window_results)} pencere)",
        )

    def reset(self):
        self._rgb_buffer.clear()
        self._time_buffer.clear()
        self._rmssd_history.clear()
        self._chrom_gain    = None
        self._gain_computed = False
