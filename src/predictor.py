# ─────────────────────────────────────────────
# TAHMİN FONKSİYONU
#
# Giriş: 5 zaman adımı, her birinde 4 değer
#   co2      → Karbondioksit (ppm)
#   temp     → Sıcaklık (°C)
#   humidity → Nem (%)
#   voc      → Uçucu organik bileşik (ppb)
#
# Çıkış:
#   0 → 🟢 Normal  — Hava kalitesi iyi
#   1 → 🟡 Dikkat  — Havalandırmayı kontrol et
#   2 → 🔴 Kritik  — Havalandırmayı DERHAL çalıştır
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd

ALERT_MESSAGES = {
    0: {
        "level":   "normal",
        "message": "Hava kalitesi iyi durumda.",
        "action":  None
    },
    1: {
        "level":   "warning",
        "message": "Dikkat! Hava kalitesi düşüyor.",
        "action":  "Havalandırma sistemini kontrol edin."
    },
    2: {
        "level":   "critical",
        "message": "KRİTİK! Hava kalitesi tehlikeli seviyede.",
        "action":  "Havalandırma sistemini DERHAL çalıştırın!"
    }
}

class Predictor:
    def __init__(self, model):
        self.model = model

    def predict_shelter(self, t0, t1, t2, t3, t4):
        X = pd.DataFrame([{
            "t0_co2": t0["co2"], "t0_temp": t0["temp"],
            "t0_humid": t0["humidity"], "t0_voc": t0["voc"],

            "t1_co2": t1["co2"], "t1_temp": t1["temp"],
            "t1_humid": t1["humidity"], "t1_voc": t1["voc"],

            "t2_co2": t2["co2"], "t2_temp": t2["temp"],
            "t2_humid": t2["humidity"], "t2_voc": t2["voc"],

            "t3_co2": t3["co2"], "t3_temp": t3["temp"],
            "t3_humid": t3["humidity"], "t3_voc": t3["voc"],

            "t4_co2": t4["co2"], "t4_temp": t4["temp"],
            "t4_humid": t4["humidity"], "t4_voc": t4["voc"],

            "co2_trend":   t4["co2"]      - t0["co2"],
            "voc_trend":   t4["voc"]      - t0["voc"],
            "temp_trend":  t4["temp"]     - t0["temp"],
            "humid_trend": t4["humidity"] - t0["humidity"],
        }])

        pred  = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0].tolist()

        co2_score   = float(np.clip(100 - (t4["co2"] - 400) / 46, 0, 100))
        voc_score   = float(np.clip(100 - (t4["voc"] - 50) / 9.5, 0, 100))
        humid_score = float(np.clip(100 - abs(t4["humidity"] - 50) * 1.8, 0, 100))
        aqi = round(co2_score * 0.5 + voc_score * 0.3 + humid_score * 0.2, 2)

        return {
            "timestamp":  pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "aqi_score":  aqi,
            "prediction": pred,
            "confidence": round(max(proba), 3),
            "alert":      ALERT_MESSAGES[pred],
            "giris": {
                "zaman1_25sn":  t0,
                "zaman2_20sn":  t1,
                "zaman3_15sn":  t2,
                "zaman4_10sn":  t3,
                "zaman5_su_an": t4
            }
        }