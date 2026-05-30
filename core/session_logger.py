"""
SessionLogger: Oturum verilerini CSV'ye kaydet.

Her satır:
timestamp, elapsed_sec, fhp_risk_score, tilt, omuz_asim, neck_var,
blink_rate, avg_ear, recent_movement, stationary_minutes,
calibration_active, risk_level, risk_score

v3 değişiklikleri:
- blink_rate, avg_ear eklendi
- recent_movement eklendi
- stationary_minutes eklendi
- flush parametre yapıldı
- exception-safe logging
- numeric rounding standardize
"""

import csv
import time
from datetime import datetime
from pathlib import Path


class SessionLogger:
    def __init__(self, log_dir: str = "logs", flush_every: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.flush_every = flush_every

        self._file = None
        self._writer = None
        self._session_start: float = 0
        self._filepath: str = ""
        self._row_count: int = 0

    # ─────────────────────────────────────────────
    # Session lifecycle
    # ─────────────────────────────────────────────

    def start_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = str(self.log_dir / f"session_{ts}.csv")
        self._session_start = time.time()
        self._row_count = 0

        self._file = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        self._writer.writerow([
            "timestamp",
            "elapsed_sec",

            "fhp_risk_score",
            "head_tilt",
            "shoulder_asym",
            "neck_variability",

            "blink_rate",
            "avg_ear",
            "recent_movement",
            "stationary_minutes",

            "calibration_active",
            "risk_level",
            "risk_score",
        ])

        self._file.flush()

    # ─────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────

    def log(self, metrics):
        if self._writer is None or metrics is None:
            return

        try:
            elapsed = round(time.time() - self._session_start, 2)

            self._writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                elapsed,

                round(metrics.fhp_risk_score, 2),
                round(metrics.head_tilt_angle, 2),
                round(metrics.shoulder_asymmetry, 4),
                round(metrics.neck_variability, 4),

                round(metrics.blink_rate, 2) if metrics.blink_rate >= 0 else -1,
                round(metrics.avg_ear, 4) if metrics.avg_ear >= 0 else -1,

                # PoseAnalyzer içinde set ediliyor
                round(getattr(metrics, "recent_movement", 0.0), 4),

                # session üzerinden alınır
                round(getattr(metrics, "stationary_minutes", 0.0), 3),

                int(metrics.calibration_active),
                metrics.risk_level,
                int(metrics.risk_score),
            ])

            self._row_count += 1

            # Flush kontrolü
            if self._row_count % self.flush_every == 0:
                self._file.flush()

        except Exception:
            # logging hiçbir zaman sistemi bozmasın
            pass

    # ─────────────────────────────────────────────
    # End session
    # ─────────────────────────────────────────────

    def end_session(self) -> str:
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            finally:
                self._file = None
                self._writer = None

        return self._filepath

    # ─────────────────────────────────────────────
    # Utils
    # ─────────────────────────────────────────────

    def get_filepath(self) -> str:
        return self._filepath

    def list_sessions(self) -> list:
        return sorted(self.log_dir.glob("session_*.csv"), reverse=True)