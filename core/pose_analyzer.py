"""
PoseAnalyzer: MediaPipe tabanlı duruş analiz motoru.
"""

import math
import time
import platform
import subprocess
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import cv2
from scipy.signal import welch
import threading


# ─── One-Euro Filter ────────────────────────────────────────────────────────

class _LowPassFilter:
    def __init__(self, alpha: float):
        self._alpha = alpha
        self._y: float = 0.0
        self._has_last: bool = False

    def update(self, x: float) -> float:
        if not self._has_last:
            self._y = x
            self._has_last = True
        else:
            self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y

    def set_alpha(self, alpha: float):
        self._alpha = alpha

    def last_value(self) -> float:
        return self._y

    def has_last_value(self) -> bool:
        return self._has_last

    def reset(self):
        self._y = 0.0
        self._has_last = False


class OneEuroFilter:
    def __init__(self, freq: float = 15.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._x = _LowPassFilter(self._alpha(min_cutoff, freq))
        self._dx = _LowPassFilter(self._alpha(dcutoff, freq))
        self._last_time: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        te = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def update(self, x: float, timestamp: Optional[float] = None) -> float:
        now = timestamp if timestamp is not None else time.monotonic()
        dt = (now - self._last_time) if self._last_time > 0 else (1.0 / 15.0)
        self._last_time = now
        if dt <= 0:
            return self._x.last_value()
        dx = (x - self._x.last_value()) / dt if self._x.has_last_value() else 0.0
        edx = self._dx.update(dx)
        cutoff = self._min_cutoff + self._beta * abs(edx)
        self._x.set_alpha(self._alpha(cutoff, 1.0 / dt))
        return self._x.update(x)

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._last_time = 0.0


# ─── Kalibrasyon ────────────────────────────────────────────────────────────

@dataclass
class CalibrationBaseline:
    eye_ratio:        float = 0.0
    nose_ratio:       float = 0.0
    ear_open:         float = 0.0   # kalibrasyondaki medyan EAR → adaptif blink threshold
    iris_px_baseline: float = 0.0
    valid:            bool  = False
    timestamp:        float = 0.0
    n_samples:        int   = 0
    just_completed:   bool  = False


# ─── Veri modelleri ──────────────────────────────────────────────────────────

@dataclass
class PostureMetrics:
    fhp_risk_score:     float
    fhp_signals:        Dict
    head_tilt_angle:    float
    shoulder_asymmetry: float
    neck_variability:   float
    nose_visibility:    float
    calibration_active: bool
    blink_rate:         float
    avg_ear:            float
    screen_distance:    float
    risk_level:         str
    risk_score:         int
    feedback:           list
    heart_rate:         float
   

    @property
    def cva_angle(self) -> float:
        return self.fhp_signals.get("eye_ratio", 0.0)

    @property
    def head_forward_ratio(self) -> float:
        return self.fhp_signals.get("nose_ratio", 0.0)


@dataclass
class SessionStats:
    total_frames:       int   = 0
    good_frames:        int   = 0
    warning_frames:     int   = 0
    critical_frames:    int   = 0
    avg_fhp_score:      float = 0.0
    avg_tilt:           float = 0.0
    stationary_minutes: float = 0.0

    @property
    def avg_cva(self) -> float:
        return self.avg_fhp_score


# ─── Eşikler ────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "fhp_score_good":     25.0,
    "fhp_score_warning":  50.0,
    "tilt_good":           5.0,
    "tilt_warning":       10.0,
    "shoulder_good":       0.03,
    "shoulder_warning":    0.06,
    "neck_var_good":       6.0,
    "neck_var_warning":    3.0,
}

NECK_VAR_WINDOW     = 900
NECK_VAR_MIN_FRAMES = 900
CALIBRATION_FRAMES  = 45

LEFT_EYE_EAR  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
EAR_BLINK_THRESH  = 0.20
EAR_CONSEC_FRAMES = 2
BLINK_WINDOW      = 900

IRIS_REAL_DIAMETER_MM  = 11.7
IRIS_BASELINE_DIST_CM  = 60.0
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
DIST_TOO_CLOSE  = 40.0    # < 40 cm kırmızı
DIST_WARN_CLOSE = 50.0    # 40-50 cm sarı
DIST_GOOD_LO    = 50.0    # 50-100 cm yeşil
DIST_GOOD_HI    = 100.0
DIST_WARN_FAR   = 100.0   # > 100 cm sarı

RPPG_BUFFER_FRAMES = 450
RPPG_UPDATE_EVERY  = 60
RPPG_BPM_LOW       = 42.0
RPPG_BPM_HIGH      = 150.0
RPPG_SNR_THRESH    = 1.0

FOREHEAD_LANDMARKS = [10, 9, 151, 108, 69, 67, 54, 21, 284, 298, 337, 338]  # sadece alın
LEFT_CHEEK_LANDMARKS  = [117, 118, 119, 120, 121, 126, 142, 203, 206, 207]   # sol yanak
RIGHT_CHEEK_LANDMARKS = [346, 347, 348, 349, 350, 355, 371, 423, 426, 427]   # sağ yanak

# ─── Ana sınıf ───────────────────────────────────────────────────────────────

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self._bpm_history = deque(maxlen=3)
        self._sitting_start: float = 0.0   # oturma başlangıç zamanı (monotonic)
        self.show_roi: bool = False


        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.session  = SessionStats()
        self.baseline = CalibrationBaseline()

        self._f_s1   = OneEuroFilter(freq=15.0, min_cutoff=1.0, beta=0.007)
        self._f_s2   = OneEuroFilter(freq=15.0, min_cutoff=0.5, beta=0.003)
        self._f_tilt = OneEuroFilter(freq=15.0, min_cutoff=1.0, beta=0.007)
        self._f_asym = OneEuroFilter(freq=15.0, min_cutoff=0.5, beta=0.003)
        self._f_fhp  = OneEuroFilter(freq=15.0, min_cutoff=0.5, beta=0.003)

        self._fhp_history:  deque = deque(maxlen=NECK_VAR_WINDOW)
        self._calib_buffer: list  = []
        self._calibrating:  bool  = False
        self.recent_movement: float = 0.0

        self._ear_history:      deque = deque(maxlen=BLINK_WINDOW)
        self._blink_frames:     int   = 0
        self._blink_count:      deque = deque(maxlen=BLINK_WINDOW)
        self._f_ear = OneEuroFilter(freq=15.0, min_cutoff=2.0, beta=0.01)

        self._iris_calib_buffer: list = []
        self._calib_ear:         list = []  # kalibrasyonda toplanan EAR değerleri
        self._f_dist = OneEuroFilter(freq=15.0, min_cutoff=0.5, beta=0.003)

        self._rppg_buffer:      deque = deque(maxlen=RPPG_BUFFER_FRAMES)
        self._rppg_frame_count: int   = 0
        self._heart_rate:       float = -1.0
        self._f_hr = OneEuroFilter(freq=1.0, min_cutoff=0.15, beta=0.05)

        self._prev_frame_time: float = 0.0
        self._current_fps:     float = 0.0

    # ─── Kalibrasyon ─────────────────────────────────────────────────────────

    def start_calibration(self):
        self._calib_buffer.clear()
        self._iris_calib_buffer.clear()
        self._calib_ear.clear()
        self._calibrating = True

    def calibration_progress(self) -> float:
        return min(len(self._calib_buffer) / CALIBRATION_FRAMES, 1.0)

    def calibration_complete(self) -> bool:
        return len(self._calib_buffer) >= CALIBRATION_FRAMES

    def consume_calibration_completed(self) -> bool:
        if self.baseline.just_completed:
            self.baseline.just_completed = False
            return True
        return False

    def _finalize_calibration(self):
        arr = np.array(self._calib_buffer)
        iris_baseline = float(np.median(self._iris_calib_buffer)) if self._iris_calib_buffer else 0.0
        self.baseline = CalibrationBaseline(
            eye_ratio        = float(np.median(arr[:, 0])),
            nose_ratio       = float(np.median(arr[:, 1])),
            ear_open         = float(np.median(self._calib_ear))  if self._calib_ear  else 0.0,
            iris_px_baseline = float(np.median(self._iris_calib_buffer)) if self._iris_calib_buffer else 0.0,
            valid          = True,
            timestamp      = time.time(),
            n_samples      = len(self._calib_buffer),
            just_completed = True,
        )
        self._calibrating = False
        self._calib_buffer.clear()
        self._iris_calib_buffer.clear()
        self._calib_ear.clear()
        self._fhp_history.clear()
        self._send_os_notification()

    def _send_os_notification(self):
        title = "PostureGuard — Kalibrasyon Tamamlandi"
        body  = "Baseline kaydedildi."
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(
                    ["osascript", "-e", f'display notification "{body}" with title "{title}"'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == "Linux":
                subprocess.Popen(
                    ["notify-send", "-t", "4000", title, body],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try:
                    from plyer import notification
                    notification.notify(title=title, message=body, timeout=4)
                except Exception:
                    pass
        except Exception:
            pass

    def reset_calibration(self):
        self.baseline = CalibrationBaseline()
        self._calibrating = False
        self._calib_buffer.clear()
        self._iris_calib_buffer.clear()
        self._calib_ear.clear()
        self._fhp_history.clear()

    # ─── Visibility gating ───────────────────────────────────────────────────

    def _landmarks_visible(self, lm) -> bool:
        """Kritik landmark'ların visibility < 0.5 ise frame'i reddet."""
        PL = self.mp_pose.PoseLandmark
        for idx in [PL.NOSE, PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER,
                    PL.LEFT_EYE, PL.RIGHT_EYE]:
            if getattr(lm[idx], "visibility", 1.0) < 0.5:
                return False
        return True

    def _should_accept_calib_sample(self, tilt: float, shoulder_asym: float, lm) -> bool:
        """Kalibrasyon kalite filtresi — tilt/asym kötüyse sample reddet."""
        if not self._landmarks_visible(lm):
            return False
        if tilt > 8.0 or shoulder_asym > 0.05:
            return False
        return True

    # ─── EAR ─────────────────────────────────────────────────────────────────

    def _blink_threshold(self) -> float:
        """Adaptif threshold: kalibrasyon EAR × 0.70. Yoksa 0.20."""
        if self.baseline.valid and self.baseline.ear_open > 0:
            return float(np.clip(self.baseline.ear_open * 0.70, 0.15, 0.25))
        return 0.20

    def _compute_ear(self, face_lm, w: int, h: int) -> Tuple[float, float, float]:
        def fp(idx):
            lm = face_lm[idx]
            return np.array([lm.x * w, lm.y * h])

        def ear_single(indices):
            p1, p2, p3, p4, p5, p6 = [fp(i) for i in indices]
            vert1 = np.linalg.norm(p2 - p6)
            vert2 = np.linalg.norm(p3 - p5)
            horiz = np.linalg.norm(p1 - p4)
            return (vert1 + vert2) / (2.0 * horiz + 1e-6)

        left_ear  = ear_single(LEFT_EYE_EAR)
        right_ear = ear_single(RIGHT_EYE_EAR)
        return (left_ear + right_ear) / 2.0, left_ear, right_ear

    def _update_blink(self, ear: float, now_mono: float) -> Tuple[float, float]:
        smooth = self._f_ear.update(ear, now_mono)
        self._ear_history.append(smooth)

        if smooth < EAR_BLINK_THRESH:
            self._blink_frames += 1
        else:
            if self._blink_frames >= EAR_CONSEC_FRAMES:
                self._blink_count.append(now_mono)
            self._blink_frames = 0

        cutoff = now_mono - 60.0
        recent_blinks = sum(1 for t in self._blink_count if t > cutoff)
        avg_ear = float(np.mean(self._ear_history)) if self._ear_history else ear
        return float(recent_blinks), avg_ear

    def _draw_ear_landmarks(self, frame, face_lm, w: int, h: int,
                             ear_val: float, blink_rate: float):
        def fp(idx):
            lm = face_lm[idx]
            return (int(lm.x * w), int(lm.y * h))

        ear_color = (0, 210, 90) if ear_val > 0.22 else (0, 165, 255) if ear_val > 0.18 else (0, 80, 255)

        for indices in [LEFT_EYE_EAR, RIGHT_EYE_EAR]:
            p1, p2, p3, p4, p5, p6 = [fp(i) for i in indices]
            cv2.line(frame, p1, p4, ear_color, 1, cv2.LINE_AA)
            cv2.line(frame, p2, p6, ear_color, 1, cv2.LINE_AA)
            cv2.line(frame, p3, p5, ear_color, 1, cv2.LINE_AA)
            for pt in [p1, p2, p3, p4, p5, p6]:
                cv2.circle(frame, pt, 2, ear_color, -1, cv2.LINE_AA)

        frame_w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame_w - 178, 0), (frame_w, 98), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.60, frame, 0.40, 0)

        blink_color = (0, 210, 90) if blink_rate >= 10 else \
                      (0, 165, 255) if blink_rate >= 5 else \
                      (180, 180, 180) if blink_rate < 0 else (0, 80, 255)
        blink_txt = "Kirpma: ..." if blink_rate < 0 else f"Kirpma: {blink_rate:.0f}/dk"

        cv2.putText(frame, f"EAR: {ear_val:.2f}",
                    (frame_w - 170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ear_color, 1, cv2.LINE_AA)
        cv2.putText(frame, blink_txt,
                    (frame_w - 170, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, blink_color, 1, cv2.LINE_AA)
        return frame

    # ─── Iris ────────────────────────────────────────────────────────────────

    def _compute_iris_px(self, face_lm, w: int, h: int) -> float:
        def fp(idx):
            lm = face_lm[idx]
            return np.array([lm.x * w, lm.y * h])

        def iris_diam(indices):
            pts = [fp(i) for i in indices[1:]]
            xs  = [p[0] for p in pts]
            return max(xs) - min(xs) if xs else 0.0

        left_d  = iris_diam(LEFT_IRIS)
        right_d = iris_diam(RIGHT_IRIS)
        avg = (left_d + right_d) / 2.0
        return avg if avg > 2.0 else -1.0

    def _compute_screen_distance(self, iris_px: float, w: int) -> float:
        if iris_px <= 0:
            return -1.0
        if self.baseline.valid and self.baseline.iris_px_baseline > 0:
            dist = IRIS_BASELINE_DIST_CM * (self.baseline.iris_px_baseline / iris_px)
        else:
            focal_px = w * 1.2
            dist_m   = (IRIS_REAL_DIAMETER_MM / 1000.0) * focal_px / iris_px
            dist     = dist_m * 100.0
        return round(float(dist), 1)

    def _draw_iris_overlay(self, frame, face_lm, w: int, h: int, dist_cm: float):
        def fp(idx):
            lm = face_lm[idx]
            return (int(lm.x * w), int(lm.y * h))

        if dist_cm < 0:
            d_color, dist_txt = (180, 180, 180), "Mesafe: ..."
        elif dist_cm < DIST_TOO_CLOSE:
            d_color, dist_txt = (0, 50, 220),  f"Mesafe: {dist_cm:.0f} cm !!"
        elif dist_cm < DIST_WARN_CLOSE:
            d_color, dist_txt = (0, 165, 255), f"Mesafe: {dist_cm:.0f} cm !"
        elif dist_cm <= DIST_GOOD_HI:
            d_color, dist_txt = (0, 210, 90),  f"Mesafe: {dist_cm:.0f} cm"
        else:
            d_color, dist_txt = (0, 165, 255), f"Mesafe: {dist_cm:.0f} cm !"

        for iris_indices in [LEFT_IRIS, RIGHT_IRIS]:
            center = fp(iris_indices[0])
            pts    = [fp(i) for i in iris_indices[1:]]
            radius = int(np.mean([
                np.linalg.norm(np.array(center) - np.array(p))
                for p in pts
            ]))
            cv2.circle(frame, center, max(radius, 3), d_color, 1, cv2.LINE_AA)
            cv2.circle(frame, center, 2, (255, 255, 255), -1, cv2.LINE_AA)

        frame_w = frame.shape[1]
        cv2.putText(frame, dist_txt,
                    (frame_w - 170, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, d_color, 1, cv2.LINE_AA)
        return frame

    # ─── rPPG CHROM ──────────────────────────────────────────────────────────



    # Değiştirilecek — _get_forehead_roi metodundaki tüm içeriği şununla değiştir:
    def _get_forehead_roi(self, face_lm, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]

        def roi_mean(landmark_list):
            pts  = np.array([[int(face_lm[i].x * w), int(face_lm[i].y * h)]
                            for i in landmark_list], dtype=np.int32)
            hull = cv2.convexHull(pts)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            pix  = frame_bgr[mask == 255]
            return pix.mean(axis=0) if len(pix) >= 30 else None

        samples = []
        for region in [FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS]:
            m = roi_mean(region)
            if m is not None:
                samples.append(m)

        if not samples:
            return None

        mean_bgr = np.mean(samples, axis=0)
        return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)
    


    
    def _chrom_bpm(self) -> float:
        """
        CHROM algoritması — Welch PSD ile BPM tahmini.
        Returns: BPM > 0, veya -1.0 (yetersiz veri), -3.0 (zayıf sinyal)
        """
        if len(self._rppg_buffer) < RPPG_BUFFER_FRAMES:
            return -1.0

        try:
            data_samples = [item[0] for item in self._rppg_buffer]
            times        = [item[1] for item in self._rppg_buffer]
            data = np.array(data_samples, dtype=np.float64)
            R, G, B = data[:, 0], data[:, 1], data[:, 2]

            # CHROM
            Rn = R / (R.mean() + 1e-6)
            Gn = G / (G.mean() + 1e-6)
            Bn = B / (B.mean() + 1e-6)
            Xs = 3 * Rn - 2 * Gn
            Ys = 1.5 * Rn + Gn - 1.5 * Bn
            S  = Xs - (Xs.std() / (Ys.std() + 1e-6)) * Ys

            # Detrend
            t_idx = np.arange(len(S))
            S = S - np.polyval(np.polyfit(t_idx, S, 1), t_idx)

            # Gerçek FPS hesapla
            duration = times[-1] - times[0]
            if duration <= 0:
                return -1.0
            real_fps = (len(times) - 1) / duration

            # Welch PSD
            nperseg = min(len(S), 128)
            freqs, psd = welch(S, fs=real_fps, nperseg=nperseg, noverlap=nperseg // 2)

            low  = RPPG_BPM_LOW  / 60.0
            high = RPPG_BPM_HIGH / 60.0
            mask = (freqs >= low) & (freqs <= high)
            
            if mask.sum() == 0:
                return -1.0

            psd_band = psd.copy()
            psd_band[~mask] = 0

            priority_mask = (freqs >= 60/60) & (freqs <= 150/60)
            if psd_band[priority_mask].size > 0 and psd_band[mask].max() > 0:
                if psd_band[priority_mask].max() > psd_band[mask].max() * 0.25:
                    psd_band[~priority_mask] = 0

            # SNR kontrolü
            peak_power  = psd_band.max()
            noise_floor = np.median(psd[mask]) + 1e-6
            snr = peak_power / noise_floor
            if snr < RPPG_SNR_THRESH:
                return -3.0

            # BPM hesapla — Gaussian ağırlıklı
            peak_idx = int(np.argmax(psd_band))
            lo = max(peak_idx - 2, 0)
            hi = min(peak_idx + 3, len(freqs))
            weights = psd_band[lo:hi]
            w_sum = weights.sum()
            if w_sum > 0:
                peak_freq = float(np.dot(freqs[lo:hi], weights) / w_sum)
            else:
                peak_freq = freqs[peak_idx]

            bpm = peak_freq * 60.0


            # Medyan smoothing
            self._bpm_history.append(bpm)
            return float(np.median(self._bpm_history))

        except Exception:
            return -1.0

    def _draw_rppg_overlay(self, frame: np.ndarray, face_lm,
                       heart_rate: float, motion_suppressed: bool) -> np.ndarray:
        
        if self.show_roi and face_lm is not None:
            h, w = frame.shape[:2]
            for region, color in [(FOREHEAD_LANDMARKS,    (0, 255, 255)),
                                (LEFT_CHEEK_LANDMARKS,  (0, 200, 200)),
                                (RIGHT_CHEEK_LANDMARKS, (0, 200, 200))]:
                pts  = np.array([[int(face_lm[i].x * w), int(face_lm[i].y * h)]
                                for i in region], dtype=np.int32)
                hull = cv2.convexHull(pts)
                cv2.polylines(frame, [hull], True, color, 1, cv2.LINE_AA)

        if motion_suppressed:
            hr_color = (180, 180, 180)
            hr_txt   = "HR: hareket"
        elif heart_rate == -3.0:
            hr_color = (0, 165, 255)
            hr_txt   = "HR: zayif sinyal"
        elif heart_rate < 0:
            buf = len(self._rppg_buffer)
            pct = int(buf / RPPG_BUFFER_FRAMES * 100)
            hr_color = (180, 180, 180)
            hr_txt   = f"HR: %{pct}"
        elif heart_rate < 60:
            hr_color = (0, 165, 255)
            hr_txt   = f"HR: {heart_rate:.0f} BPM"
        elif heart_rate <= 100:
            hr_color = (0, 210, 90)
            hr_txt   = f"HR: {heart_rate:.0f} BPM"
        else:
            hr_color = (0, 80, 255)
            hr_txt   = f"HR: {heart_rate:.0f} BPM"

        frame_w = frame.shape[1]
        cv2.putText(frame, hr_txt,
                    (frame_w - 170, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, hr_color, 1, cv2.LINE_AA)
        return frame

    # ─── Sinyal hesaplama ─────────────────────────────────────────────────────

    def _compute_signals(self, lm, w: int, h: int) -> Dict:
        PL = self.mp_pose.PoseLandmark

        def pt(idx):
            l = lm[idx]
            return np.array([l.x * w, l.y * h])

        nose           = pt(PL.NOSE)
        left_shoulder  = pt(PL.LEFT_SHOULDER)
        right_shoulder = pt(PL.RIGHT_SHOULDER)
        left_eye       = pt(PL.LEFT_EYE)
        right_eye      = pt(PL.RIGHT_EYE)
        left_ear       = pt(PL.LEFT_EAR)
        right_ear      = pt(PL.RIGHT_EAR)

        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        mid_eye      = (left_eye + right_eye) / 2.0
        mid_ear      = (left_ear + right_ear) / 2.0
        shoulder_w   = abs(left_shoulder[0] - right_shoulder[0]) + 1e-6

        s1 = (mid_shoulder[1] - mid_eye[1])  / shoulder_w
        s2 = (mid_shoulder[1] - nose[1])     / shoulder_w

        eye_vec  = right_eye - left_eye
        tilt_raw = abs(math.degrees(math.atan2(
            abs(float(eye_vec[1])),
            abs(float(eye_vec[0])) + 1e-6
        )))

        return {
            "eye_ratio":    s1,
            "nose_ratio":   s2,
            "tilt_raw":     tilt_raw,
            "mid_shoulder": mid_shoulder,
            "mid_ear":      mid_ear,
        }

    # ─── Skor ────────────────────────────────────────────────────────────────

    def _composite_score(self, signals: Dict) -> float:
        s1 = max(0.0, (1.5 - signals["eye_ratio"])  / 1.0) * 50
        s2 = max(0.0, (1.2 - signals["nose_ratio"]) / 0.8) * 50
        return min((s1 + s2) / 2.0, 100.0)

    def _deviation_score(self, signals: Dict) -> float:
        d_eye  = max(self.baseline.eye_ratio  - signals["eye_ratio"],  0.0)
        d_nose = max(self.baseline.nose_ratio - signals["nose_ratio"], 0.0)
        score  = min((d_eye  / 0.15) * 40, 40.0)
        score += min((d_nose / 0.12) * 60, 60.0)
        return min(score, 100.0)

    # ─── Ana analiz ──────────────────────────────────────────────────────────

    def analyze_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[PostureMetrics]]:
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(rgb)
        face_results = self.face_mesh.process(rgb)

        if not pose_results.pose_landmarks:
            return self._annotate_no_detection(frame_bgr), None

        lm  = pose_results.pose_landmarks.landmark
        PL  = self.mp_pose.PoseLandmark
        now = time.monotonic()

        # Visibility gating — güvenilmez frame'lerde ölçüm yapma
        if not self._landmarks_visible(lm):
            return self._annotate_no_detection(frame_bgr), None

        # FPS
        if self._prev_frame_time > 0:
            dt = now - self._prev_frame_time
            raw_fps = 1.0 / dt if dt > 0 else 0.0
            self._current_fps = self._current_fps * 0.9 + raw_fps * 0.1 if self._current_fps > 0 else raw_fps
        self._prev_frame_time = now

        raw = self._compute_signals(lm, w, h)

        eye_s  = self._f_s1.update(raw["eye_ratio"],  now)
        nose_s = self._f_s2.update(raw["nose_ratio"], now)
        tilt_s = self._f_tilt.update(raw["tilt_raw"], now)

        signals = {
            "eye_ratio":    eye_s,
            "nose_ratio":   nose_s,
            "mid_shoulder": raw["mid_shoulder"],
            "mid_ear":      raw["mid_ear"],
        }

        if self.baseline.valid:
            fhp_score    = self._deviation_score(signals)
            calib_active = True
        else:
            fhp_score    = self._composite_score(signals)
            calib_active = False

        smooth_fhp = self._f_fhp.update(fhp_score, now)

        def pt(idx):
            l = lm[idx]
            return np.array([l.x * w, l.y * h])
        ls = pt(PL.LEFT_SHOULDER)
        rs = pt(PL.RIGHT_SHOULDER)
        smooth_asym = self._f_asym.update(abs(ls[1] - rs[1]) / h, now)

        SHORT_WINDOW = 75
        self._fhp_history.append(eye_s)

        if len(self._fhp_history) >= NECK_VAR_MIN_FRAMES:
            window_sec = len(self._fhp_history) / 15.0
            raw_std    = float(np.std(self._fhp_history))
            neck_var   = -1.0 if raw_std < 0.003 else raw_std * (60.0 / window_sec)
        else:
            neck_var = -1.0

        if len(self._fhp_history) >= SHORT_WINDOW:
            self.recent_movement = float(np.std(list(self._fhp_history)[-SHORT_WINDOW:]))
        else:
            self.recent_movement = 0.0

        if face_results and face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0].landmark

            ear_raw, _, _ = self._compute_ear(face_lm, w, h)
            blink_rate, avg_ear = self._update_blink(ear_raw, now)

            iris_px = self._compute_iris_px(face_lm, w, h)
            raw_dist = self._compute_screen_distance(iris_px, w)
            screen_distance = self._f_dist.update(raw_dist, now) if raw_dist > 0 else -1.0

            motion_suppressed = self.recent_movement > 0.01
            if not motion_suppressed:
                rgb_sample = self._get_forehead_roi(face_lm, frame_bgr)
                if rgb_sample is not None:
                    self._rppg_buffer.append((rgb_sample, now))

            self._rppg_frame_count += 1
            if self._rppg_frame_count >= RPPG_UPDATE_EVERY:
                self._rppg_frame_count = 0
                if not motion_suppressed:
                    bpm = self._chrom_bpm()
                    if bpm > 0:
                        self._heart_rate = round(self._f_hr.update(bpm, now), 1)
                    elif bpm == -3.0:
                        self._heart_rate = -3.0

            heart_rate = -2.0 if motion_suppressed else self._heart_rate
        else:
            face_lm           = None
            blink_rate        = -1.0
            avg_ear           = -1.0
            screen_distance   = -1.0
            heart_rate        = -1.0
            motion_suppressed = False
            self._rppg_buffer.clear()
            self._rppg_frame_count = 0
            self._heart_rate = -1.0
            iris_px = -1.0
            avg_ear = -1.0
            self._sitting_start = 0.0
            self.session.stationary_minutes = 0.0

        # Kalibrasyon buffer — FaceMesh sonrası (avg_ear, iris_px tanımlı)
        if self._calibrating:
            if self._should_accept_calib_sample(tilt_s, smooth_asym, lm):
                self._calib_buffer.append([eye_s, nose_s])
                if face_lm is not None and avg_ear > 0:
                    self._calib_ear.append(float(avg_ear))
                if face_lm is not None and iris_px > 0:
                    self._iris_calib_buffer.append(iris_px)
            if self.calibration_complete():
                self._finalize_calibration()

        metrics = self._classify(
            fhp_score       = smooth_fhp,
            signals         = signals,
            tilt            = tilt_s,
            shoulder_asym   = smooth_asym,
            neck_var        = neck_var,
            nose_vis        = lm[PL.NOSE].visibility,
            calib_active    = calib_active,
            blink_rate      = blink_rate,
            avg_ear         = avg_ear,
            screen_distance = screen_distance,
            heart_rate      = heart_rate,
        )

        self._update_session(metrics)

        annotated = self._draw_overlay(
            frame_bgr.copy(), pose_results,
            raw["mid_shoulder"], raw["mid_ear"],
            smooth_fhp, tilt_s, metrics.risk_level, calib_active
        )

        if face_lm is not None:
            annotated = self._draw_ear_landmarks(annotated, face_lm, w, h, avg_ear, blink_rate)
            annotated = self._draw_iris_overlay(annotated, face_lm, w, h, screen_distance)
            annotated = self._draw_rppg_overlay(annotated, face_lm, heart_rate, motion_suppressed)

        return annotated, metrics

    # ─── Risk sınıflandırma ──────────────────────────────────────────────────

    def _classify(self, fhp_score, signals, tilt, shoulder_asym,
                  neck_var, nose_vis, calib_active,
                  blink_rate, avg_ear, screen_distance, heart_rate) -> PostureMetrics:
        T        = THRESHOLDS
        score    = 0
        feedback = []

        if fhp_score >= T["fhp_score_warning"]:
            score += 40
            feedback.append("Bas baseline'indan belirgin one egilmis" if calib_active
                            else "Bas belirgin sekilde one egilmis")
        elif fhp_score >= T["fhp_score_good"]:
            score += 20
            feedback.append("Hafif one egilme var — omurganı dik tut")

        if tilt > T["tilt_warning"]:
            score += 30
            feedback.append("Bas yana egik — basini dik tut")
        elif tilt > T["tilt_good"]:
            score += 15
            feedback.append("Hafif bas egikligi var")

        if shoulder_asym > T["shoulder_warning"]:
            score += 20
            feedback.append("Omuzlar esit degil — klavye/mouse pozisyonunu kontrol et")
        elif shoulder_asym > T["shoulder_good"]:
            score += 10
            feedback.append("Hafif omuz yükseklik farki")

        if blink_rate >= 0 and (len(self._blink_count) > 0 or blink_rate == 0):
            if 0 <= blink_rate < 5:
                score += 10
                feedback.append("Göz kirpma cok az — CVS riski")
            elif 5 <= blink_rate < 10:
                feedback.append("Göz kirpma normale gore az")

        if screen_distance > 0:
            if screen_distance < DIST_TOO_CLOSE:
                score += 20
                feedback.append(f"Ekrana cok yakınsin ({screen_distance:.0f} cm) — min 40 cm")
            elif screen_distance < DIST_WARN_CLOSE:
                score += 10
                feedback.append(f"Ekran biraz yakın ({screen_distance:.0f} cm) — ideal 50-100 cm")
            elif screen_distance > DIST_WARN_FAR:
                score += 5
                feedback.append(f"Ekran cok uzak ({screen_distance:.0f} cm) — one egilme riski")

        if nose_vis < 0.5:
            feedback.append("Yuz net gorunmuyor — kamera acisini ayarla")

        # Yeni — hemen altına ekle:
        if shoulder_asym > T["shoulder_warning"] * 3:
            feedback.append("Omuz tespiti zayif — acik renk giysi/arka plan onerilir")

        if not feedback:
            feedback.append("Duruş baseline'a yakın — böyle devam et" if calib_active
                            else "Duruş iyi — böyle devam et")

        risk = "critical" if score >= 50 else "warning" if score >= 20 else "good"

        return PostureMetrics(
            fhp_risk_score    = round(fhp_score, 1),
            fhp_signals       = {
                "eye_ratio":  round(signals["eye_ratio"],  3),
                "nose_ratio": round(signals["nose_ratio"], 3),
            },
            head_tilt_angle    = round(tilt, 1),
            shoulder_asymmetry = round(shoulder_asym, 4),
            neck_variability   = round(neck_var, 2),
            nose_visibility    = round(nose_vis, 2),
            calibration_active = calib_active,
            blink_rate         = round(blink_rate, 1),
            avg_ear            = round(avg_ear, 3),
            screen_distance    = round(screen_distance, 1),
            heart_rate         = round(heart_rate, 1),
            risk_level         = risk,
            risk_score         = min(score, 100),
            feedback           = feedback,
        )

    # ─── Oturum ──────────────────────────────────────────────────────────────

    def _update_session(self, metrics: PostureMetrics):
        s = self.session
        s.total_frames += 1
        if metrics.risk_level == "good":      s.good_frames += 1
        elif metrics.risk_level == "warning": s.warning_frames += 1
        else:                                 s.critical_frames += 1
        n = s.total_frames
        s.avg_fhp_score = (s.avg_fhp_score * (n-1) + metrics.fhp_risk_score) / n
        s.avg_tilt      = (s.avg_tilt      * (n-1) + metrics.head_tilt_angle) / n

        # Toplam oturma süresi — yüz algılanıyorsa say
        now = time.monotonic()
        if self._sitting_start == 0.0:
            self._sitting_start = now
        s.stationary_minutes = (now - self._sitting_start) / 60.0

    def get_posture_score(self) -> float:
        if self.session.total_frames == 0:
            return 100.0
        return round(self.session.good_frames / self.session.total_frames * 100, 1)

    def reset_session(self):
        self.session = SessionStats()
        self._f_s1.reset(); self._f_s2.reset()
        self._f_tilt.reset(); self._f_asym.reset()
        self._f_fhp.reset(); self._f_ear.reset()
        self._f_dist.reset(); self._f_hr.reset()
        self._fhp_history.clear()
        self._ear_history.clear()
        self._blink_count.clear()
        self._blink_frames     = 0
        self._rppg_buffer.clear()
        self._rppg_frame_count = 0
        self._heart_rate       = -1.0
        self._bpm_history.clear()
        self._prev_frame_time  = 0.0
        self._current_fps      = 0.0
        self._sitting_start = 0.0
        self.reset_calibration()

    # ─── Görselleştirme ──────────────────────────────────────────────────────

    def _draw_overlay(self, frame, pose_results, mid_shoulder, mid_ear,
                      fhp_score, tilt, risk_level, calib_active):
        color_map = {
            "good":     (0, 210, 90),
            "warning":  (0, 165, 255),
            "critical": (0, 50, 220),
        }
        color = color_map[risk_level]

        PL = self.mp_pose.PoseLandmark
        lm = pose_results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        def pt(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        ls, rs = pt(PL.LEFT_SHOULDER), pt(PL.RIGHT_SHOULDER)
        cv2.line(frame, ls, rs, color, 2, cv2.LINE_AA)
        cv2.circle(frame, ls, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, rs, 4, color, -1, cv2.LINE_AA)
        cv2.line(frame, tuple(mid_ear.astype(int)), tuple(mid_shoulder.astype(int)),
                 color, 2, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, 75), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.60, frame, 0.40, 0)

        tag = "CAL" if calib_active else "EST"
        cv2.putText(frame, f"FHP {fhp_score:.0f}  [{tag}]",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Risk: {risk_level.upper()}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self._current_fps:.1f}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        return frame

    def _annotate_no_detection(self, frame):
        cv2.putText(frame, "Yuz/Vucut algilanamadi",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pose.close()
        self.face_mesh.close()

    def __del__(self):
        try:
            self.pose.close()
            self.face_mesh.close()
        except Exception:
            pass