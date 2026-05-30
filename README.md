# 🖥️ DeskPoseAI

**Gerçek Zamanlı Masabaşı Ergonomi İzleme**  
Ofis çalışanları için kamera tabanlı ergonomi analizi. Tek bir dizüstü bilgisayar web kamerası ile çalışır — ek donanım gerekmez.

![DeskPoseAI Genel Bakış](docs/overview_en.png)

[![Streamlit Uygulaması](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deskposeai.streamlit.app)

---

## Özellikler

### Duruş Analizi
- **FHP Risk Skoru** — Baş Öne Eğilme (Forward Head Posture) riski, 0–100 (göz-omuz / omuz genişliği oranı)
- **Kalibre Mod (CAL)** — Kişisel baseline kaydedilir, sapma skoru hesaplanır
- **Tahmini Mod (EST)** — Kalibrasyon olmadan mutlak referans değerleri
- **Baş Eğim Açısı (Roll)** — `arctan(Δy/Δx)` — yanal fleksiyon tespiti
- **Omuz Asimetrisi** — Sol/sağ omuz yükseklik farkı (trapezius gerilim göstergesi)

### Göz Sağlığı
- **EAR Göz Kırpma Hızı** — Göz Görünüm Oranı / Eye Aspect Ratio (Soukupová & Čech 2016)
  - Adaptif eşik: kalibrasyon EAR × 0.70
  - Normal: 15–20 kırpma/dk, ekran başında 3–7/dk'ya düşer
  - CVS (Bilgisayar Görme Sendromu) erken uyarısı
- **Ekran Mesafesi** — Iris çapı tabanlı (Bekerman 2014, iris = 11,7 mm sabit)
  - Iris Çapı Sabiti | Jonuscheit ve ark., Ophthalmic Physiol Opt 2019 — HVID ortalama ~11,8 mm |
  - Kalibrasyonlu: baseline iris pikseli × mesafe oranı
  - Kalibrasyonsuz: delik göz buluşu (~60° GGA)
  - ISO 9241 standardı: 50–100 cm ideal

### Yaşamsal Bulgular
- **Kalp Atış Hızı (rPPG)** — CHROM algoritması (De Haan & Jeanne 2013)
  - Alın + sol yanak + sağ yanak ROI
  - Welch PSD frekans analizi
  - Fizyolojik öncelik bandı: 60–150 BPM
  - SNR kontrolü, medyan yumuşatma
  - ROI görselleştirme açma/kapama

### Ergonomi Takibi
- **Oturma Süresi** — Yüz algılandığı sürece sayar, kullanıcı ayrılınca sıfırlanır
  - 30 dk → mola hatırlatıcısı
  - 60 dk → ayağa kalk uyarısı
  - 90 dk → uzun mola gerekli
- **Kalibrasyon Kalite Filtresi** — Eğim/asimetri bozuksa örnek reddedilir
- **Görünürlük Kapısı** — Güvenilmez kareler iptal edilir

---

## Algoritmalar

### FHP (Baş Öne Eğilme)

```
S1 = (omuz_y - göz_y)  / omuz_genişliği   → öne eğilince azalır
S2 = (omuz_y - burun_y) / omuz_genişliği  → daha hassas sinyal

CAL modu: delta_göz × 40 + delta_burun × 60 → sapma skoru
EST modu: (1.5 - S1) × 50 + (1.2 - S2) × 50 → bileşik skor
```

### EAR (Göz Görünüm Oranı)

```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
Kırpma: EAR < eşik, ≥2 ardışık kare
Adaptif eşik: kalibrasyon_EAR × 0.70 (sınır: 0.15–0.25)
```

### Ekran Mesafesi

```
Kalibrasyonlu:    mesafe = 60 cm × (baseline_iris_piksel / mevcut_iris_piksel)
Kalibrasyonsuz:   mesafe = (11.7 mm / 1000) × odak_piksel / iris_piksel × 100
```

### rPPG CHROM

```
1. Alın + yanak ROI → RGB ortalama (hareket algılanırsa atla)
2. Normalleştir: Rn = R/ort(R), Gn, Bn
3. Krominans: Xs = 3Rn - 2Gn, Ys = 1.5Rn + Gn - 1.5Bn
4. S = Xs - (std_Xs / std_Ys) × Ys → trend giderme
5. Welch PSD (nperseg=128, noverlap=64)
6. Öncelik bandı: 60–150 BPM
7. SNR = tepe / medyan(gürültü) > 1.0
8. Gauss ağırlıklı tepe → BPM
9. Medyan yumuşatma (son 15 değer)
```

---

## Sinyal Filtreleme

**One-Euro Filtresi** — tüm sinyallere uygulanır:
- Düşük hızda yüksek yumuşatma, yüksek hızda düşük yumuşatma
- Gecikmeyi minimize eder

**Parametreler:**

| Sinyal | min_cutoff | beta |
|--------|-----------|------|
| FHP (S1, S2) | 1.0 / 0.5 | 0.007 / 0.003 |
| Eğim | 1.0 | 0.007 |
| Omuz asim. | 0.5 | 0.003 |
| EAR | 2.0 | 0.01 |
| Ekran mesafesi | 0.5 | 0.003 |
| Kalp atış hızı | 0.15 | 0.05 |

---

## Risk Eşikleri

| Metrik | 🟢 İyi | 🟡 Uyarı | 🔴 Kritik |
|--------|--------|----------|----------|
| FHP Skoru | < 25 | 25–50 | > 50 |
| Baş eğimi | < 5° | 5–10° | > 10° |
| Omuz asim. | < %3 | %3–6 | > %6 |
| Göz kırpma | ≥ 10/dk | 5–10/dk | < 5/dk |
| Ekran mesafesi | 50–100 cm | 40–50 cm | < 40 cm |
| Oturma süresi | < 30 dk | 30–60 dk | > 60 dk |
| Kalp atış hızı | 60–100 BPM | 50–60 / 100–120 | < 50 / > 120 |

---

## Kurulum

```bash
git clone https://github.com/Ugeyik83/DeskPoseAI.git
cd DeskPoseAI
pip install -r requirements.txt
streamlit run app.py
```

### Gereksinimler

```
streamlit
streamlit-webrtc
mediapipe
opencv-python
numpy
scipy
av
```

---

## Kullanım

1. **START** — kamerayı etkinleştir
2. **Kalibre Et** — dik otur, 3 sn bekle → kişisel baseline kaydedilir
3. Metrik kartlarını izle
4. **CAL** rozeti → kalibre mod aktif
5. **EST** rozeti → tahmini mod (kalibrasyon önerilir)

### Kalibrasyon ipuçları
- Dik otur, omuzların seviyede olsun
- Doğrudan kameraya bak
- 3 saniye hareketsiz kal
- İyi aydınlatma → daha iyi rPPG doğruluğu

---

## Sınırlamalar

| Kısıt | Açıklama |
|-------|----------|
| **Yalnızca ön kamera** | CVA ölçümü klinik değil, yaklaşımdır |
| **rPPG doğruluğu** | ±5–8 BPM, ışık ve harekete bağımlı |
| **Ekran mesafesi** | Kalibrasyonsuz ±10–15 cm hata |
| **Koyu giysi** | Omuz tespiti bozulur |
| **Yetersiz aydınlatma** | rPPG güvenilmez |

> FHP Risk Skoru klinik bir CVA ölçümü değildir. Tıbbi amaçla kullanılamaz.

---

## Dosya Yapısı

```
DeskPoseAI/
├── app.py                 # Streamlit UI + WebRTC
├── requirements.txt
├── packages.txt
├── core/
│   ├── pose_analyzer.py   # Ana analiz motoru
│   ├── alert_manager.py   # İşletim sistemi bildirimleri
│   └── session_logger.py  # CSV kayıt
└── logs/                  # Oturum kayıtları
```

---

## Yol Haritası

### Kısa Vadeli
- [ ] **Solunum Hızı** — omuz y + rPPG düşük bant füzyonu (doğrulama cihazı gerekli: Polar H10)
- [ ] **3D Baş Pozu (solvePnP)** — FaceMesh + OpenCV gerçek pitch/yaw/roll → daha doğru CVA yaklaşımı
- [ ] **20-20-20 Hatırlatıcısı** — 20 dk çalış → 20 sn uzağa bak → göz yorgunluğu önleme
- [ ] **Oturum PDF Raporu** — ergonomi özeti, trend grafikleri, öneriler

### Orta Vadeli
- [ ] **Masaüstü Uygulaması** — CustomTkinter + `cv2.VideoCapture` → WebRTC bağımlılığını kaldır
- [ ] **PyInstaller Paketi** — Windows `.exe` / macOS `.app` → Python gerekmez
- [ ] **IPD Füzyonu** — Iris + gözlerarası mesafe → ekran mesafesi hatası ±3 cm
- [ ] **Çok Kullanıcılı Profiller** — Kullanıcı başına ayrı baseline

### Uzun Vadeli
- [ ] **ML Tabanlı FHP** — Landmark'lardan Random Forest/LSTM duruş sınıflandırması
- [ ] **Ergonomi Skoru Trendleri** — Günlük/haftalık istatistik panosu
- [ ] **İSG Entegrasyonu** — Kurumsal çalışan sağlığı yönetim sistemi API
- [ ] **Egzoskeleton Tetikleyici** — Kritik duruş → egzoskeleton aktivasyon sinyali (araştırma)
- [ ] **HRV** — Yüksek FPS kamera + kontrollü ortamda yeniden dene

---

## Kaynaklar

| Algoritma | Kaynak |
|-----------|--------|
| EAR Göz Kırpma Tespiti | Soukupová & Čech, BMVC 2016 |
| CHROM rPPG | De Haan & Jeanne, IEEE TBME 2013 |
| Iris Çapı Sabiti | Bekerman ve ark., IOVS 2014 |
| CVA Klinik Eşiği | Griegel-Morris ve ark., PTJ 1992 |
| Ekran Mesafesi Standardı | ISO 9241-5 |
| One-Euro Filtresi | Casiez ve ark., CHI 2012 |

---

## Lisans

MIT Lisansı — akademik ve ticari kullanıma açık.