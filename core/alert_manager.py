"""
AlertManager: OS bildirim yönetimi.
Windows/macOS/Linux çapraz platform destek.
Cooldown mantığı ile spam koruması.

v2 iyileştirmeleri:
- warning/critical için ayrı consecutive sayaç (daha doğru eşik)
- global_min_interval ile ek spam koruması
- risk seviyesi iyiye dönünce decay (tam sıfırlamak yerine) -> daha stabil
- opsiyonel "stationary" kanal desteği (hareketsizlik uyarısı)
- kalibrasyon sırasında bildirim kapatma (default)
- notifier tespiti: shutil.which + daha güvenilir fallback zinciri
"""

import time
import threading
import platform
import subprocess
import re
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


@dataclass
class AlertConfig:
    # Cooldown
    cooldown_sec: int = 30             # warning tekrar aralığı
    critical_cooldown: int = 20        # critical tekrar aralığı
    stationary_cooldown: int = 120     # hareketsizlik tekrar aralığı

    # Eşikler (ardışık frame)
    warning_threshold: int = 5
    critical_threshold: int = 3
    stationary_threshold_sec: int = 60  # hareketsizlik (saniye) bazlı tetikleme (opsiyonel)

    # Ek spam koruması
    global_min_interval: int = 8       # seviyeden bağımsız minimum interval

    # Davranış
    decay_on_good: bool = True         # good gelince streak'i yumuşatarak azalt
    decay_step: int = 2                # good frame -> streak azaltma miktarı
    disable_during_calibration: bool = True  # kalibrasyonda bildirim gönderme


class AlertManager:
    """
    process(...) çağrısını her frame'de yapabilirsin (thread-safe).
    """
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()

        self._lock = threading.Lock()
        self._platform = platform.system()  # Windows / Darwin / Linux
        self._notifier = self._detect_notifier()

        # last alert timestamps
        self._last_alert_time: Dict[str, float] = {
            "warning": 0.0,
            "critical": 0.0,
            "stationary": 0.0,
            "global": 0.0,
        }

        # separate streak counters
        self._streak_warning = 0
        self._streak_critical = 0

        # hareketsizlik için (opsiyonel)
        self._stationary_start_ts: float = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Notifier detection
    # ─────────────────────────────────────────────────────────────────────

    def _detect_notifier(self) -> str:
        """
        Mevcut bildirim sistemini tespit et.
        Return: "win10toast" | "plyer" | "powershell" | "osascript" | "notify-send" | "none"
        """
        if self._platform == "Windows":
            # win10toast varsa onu kullan
            try:
                import win10toast  # noqa: F401
                return "win10toast"
            except Exception:
                # plyer çoğu ortamda iş görür
                try:
                    import plyer  # noqa: F401
                    return "plyer"
                except Exception:
                    return "powershell"

        if self._platform == "Darwin":
            return "osascript"

        # Linux
        if shutil.which("notify-send"):
            return "notify-send"

        # fallback
        try:
            import plyer  # noqa: F401
            return "plyer"
        except Exception:
            return "none"

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def reset(self):
        with self._lock:
            for k in self._last_alert_time:
                self._last_alert_time[k] = 0.0
            self._streak_warning = 0
            self._streak_critical = 0
            self._stationary_start_ts = 0.0

    def process(
        self,
        risk_level: str,
        feedback: List[str],
        *,
        calibration_active: bool = False,
        stationary_minutes: Optional[float] = None,
        recent_movement: Optional[float] = None,
    ) -> bool:
        """
        Risk seviyesine göre bildirim mantığı.
        Returns: True if a notification was sent.

        Params:
          - calibration_active: kalibrasyon sırasında disable edilebilir
          - stationary_minutes: opsiyonel hareketsizlik alarmı için (PoseAnalyzer session)
          - recent_movement: opsiyonel (ileride daha akıllı bastırma için)
        """
        with self._lock:
            now = time.time()

            # kalibrasyonda bildirim istemiyorsak çık
            if self.config.disable_during_calibration and calibration_active:
                self._decay_on_good()  # kalibrasyon esnasında da streak şişmesin
                return False

            # global spam guard
            if now - self._last_alert_time["global"] < self.config.global_min_interval:
                # yine de streakleri güncelleyelim
                self._update_streaks(risk_level)
                return False

            # stationary opsiyonel
            sent_stationary = False
            if stationary_minutes is not None:
                sent_stationary = self._process_stationary(now, stationary_minutes)

            # main posture alert
            sent_posture = self._process_posture(now, risk_level, feedback)

            return bool(sent_stationary or sent_posture)

    def seconds_until_next_alert(self, level: str) -> float:
        """Bir sonraki bildirime kaç saniye kaldı."""
        with self._lock:
            now = time.time()
            cd = self._cooldown_for(level)
            elapsed = now - self._last_alert_time.get(level, 0.0)
            return max(0.0, cd - elapsed)

    # ─────────────────────────────────────────────────────────────────────
    # Internal: posture alerts
    # ─────────────────────────────────────────────────────────────────────

    def _process_posture(self, now: float, risk_level: str, feedback: List[str]) -> bool:
        # Streakleri güncelle
        self._update_streaks(risk_level)

        if risk_level == "good":
            # good ise bildirim yok
            return False

        # threshold check
        if risk_level == "critical":
            if self._streak_critical < self.config.critical_threshold:
                return False
            level = "critical"
        else:
            if self._streak_warning < self.config.warning_threshold:
                return False
            level = "warning"

        # cooldown check (level-specific)
        cooldown = self._cooldown_for(level)
        if now - self._last_alert_time.get(level, 0.0) < cooldown:
            return False

        title, body = self._build_message(level, feedback)
        sent = self._send_notification(title, body, level)

        if sent:
            self._last_alert_time[level] = now
            self._last_alert_time["global"] = now
            # after sending, reset streak for that level only
            if level == "critical":
                self._streak_critical = 0
            else:
                self._streak_warning = 0

        return sent

    def _update_streaks(self, risk_level: str):
        if risk_level == "critical":
            self._streak_critical += 1
            # critical aynı zamanda warning'ı da kötü sayar
            self._streak_warning += 1
        elif risk_level == "warning":
            self._streak_warning += 1
            # critical streak'i yavaşça erisin (ya da sıfırla)
            if self.config.decay_on_good:
                self._streak_critical = max(0, self._streak_critical - 1)
            else:
                self._streak_critical = 0
        else:
            # good
            self._decay_on_good()

    def _decay_on_good(self):
        if self.config.decay_on_good:
            self._streak_warning = max(0, self._streak_warning - self.config.decay_step)
            self._streak_critical = max(0, self._streak_critical - self.config.decay_step)
        else:
            self._streak_warning = 0
            self._streak_critical = 0

    # ─────────────────────────────────────────────────────────────────────
    # Internal: stationary alerts (optional)
    # ─────────────────────────────────────────────────────────────────────

    def _process_stationary(self, now: float, stationary_minutes: float) -> bool:
        """
        stationary_minutes PoseAnalyzer.session.stationary_minutes üzerinden gelir.
        1 dk üstü gibi bir eşik konulabilir.
        """
        # saniye bazlı eşik
        stationary_sec = stationary_minutes * 60.0

        # eşik altındaysa reset
        if stationary_sec < float(self.config.stationary_threshold_sec):
            self._stationary_start_ts = 0.0
            return False

        # cooldown
        if now - self._last_alert_time.get("stationary", 0.0) < self.config.stationary_cooldown:
            return False

        title = "🧍 PostureGuard — Hareketsizlik"
        body = f"{stationary_minutes:.1f} dk hareketsizsin. Kısa bir mola/esneme iyi gelir."

        sent = self._send_notification(title, body, "stationary")
        if sent:
            self._last_alert_time["stationary"] = now
            self._last_alert_time["global"] = now
        return sent

    # ─────────────────────────────────────────────────────────────────────
    # Message building / sanitization
    # ─────────────────────────────────────────────────────────────────────

    def _build_message(self, level: str, feedback: List[str]) -> Tuple[str, str]:
        if level == "critical":
            title = "🚨 PostureGuard — Duruş Kritik!"
        else:
            title = "⚡ PostureGuard — Duruş Uyarısı"

        clean = self._sanitize_feedback(feedback)
        body = "\n".join(clean[:2]) if clean else "Duruşunu kontrol et."
        return title, body

    def _sanitize_feedback(self, feedback: List[str]) -> List[str]:
        if not feedback:
            return []
        out = []
        for f in feedback:
            if not isinstance(f, str):
                continue
            # emoji + ikonları temizle (başlangıç)
            f = re.sub(r"^[^\w\d]+", "", f).strip()
            # çift boşlukları toparla
            f = re.sub(r"\s+", " ", f).strip()
            if f:
                out.append(f)
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Sending notifications
    # ─────────────────────────────────────────────────────────────────────

    def _cooldown_for(self, level: str) -> int:
        if level == "critical":
            return int(self.config.critical_cooldown)
        if level == "stationary":
            return int(self.config.stationary_cooldown)
        return int(self.config.cooldown_sec)

    def _send_notification(self, title: str, body: str, level: str) -> bool:
        try:
            if self._notifier == "win10toast":
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(title, body, duration=5, threaded=True)
                return True

            if self._notifier == "plyer":
                from plyer import notification
                notification.notify(
                    title=title,
                    message=body,
                    app_name="PostureGuard",
                    timeout=5
                )
                return True

            if self._notifier == "powershell":
                # basit PS toast (çok minimal). En azından kullanıcı görür.
                # Not: bazı ortamlar izin vermeyebilir; hata yakalanır.
                ps = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template)
$xml.GetElementsByTagName("text")[0].AppendChild($xml.CreateTextNode("{title}")) > $null
$xml.GetElementsByTagName("text")[1].AppendChild($xml.CreateTextNode("{body}")) > $null
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("PostureGuard").Show($toast)
"""
                subprocess.Popen(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True

            if self._notifier == "osascript":
                script = f'display notification "{body}" with title "{title}"'
                subprocess.Popen(
                    ["osascript", "-e", script],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True

            if self._notifier == "notify-send":
                urgency = "critical" if level == "critical" else "normal"
                subprocess.Popen(
                    ["notify-send", "-u", urgency, "-t", "5000", title, body],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True

            return False

        except Exception as e:
            print(f"[AlertManager] Bildirim gönderilemedi: {e}")
            return False
