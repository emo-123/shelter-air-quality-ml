# ─────────────────────────────────────────────
# ANA ÇALIŞTIRMA DOSYASI
# Tüm sınıfları bir araya getirir ve
# manuel test arayüzünü çalıştırır.
# ─────────────────────────────────────────────

import os
import json
import warnings
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader  import DataLoader
from src.preprocessor import Preprocessor
from src.trainer      import Trainer
from src.predictor    import Predictor

warnings.filterwarnings("ignore")

def manuel_giris(zaman_adi):
    print(f"── {zaman_adi} ──")
    co2      = float(input("  CO2 (ppm)     : "))
    temp     = float(input("  Sıcaklık (°C) : "))
    humidity = float(input("  Nem (%)       : "))
    voc      = float(input("  VOC (ppb)     : "))
    return {"co2": co2, "temp": temp, "humidity": humidity, "voc": voc}

if __name__ == "__main__":

    # ── 1. Veri yükle ──
    loader = DataLoader("data/shelter_sensor_dataset_v2.csv")
    df = loader.load()

    # ── 2. Temizle ve AQI ekle ──
    prep = Preprocessor()
    df = prep.clean(df)
    df = prep.add_aqi(df)

    # ── 3. Model veri seti oluştur ──
    model_df = prep.build_model_dataset(df)

    # ── 4. Modeli eğit ──
    trainer = Trainer()
    model, X_train = trainer.train(model_df)
    trainer.plot_feature_importance(model, X_train)

    # ── 5. Manuel test arayüzü ──
    predictor = Predictor(model)

    print("\n" + "=" * 60)
    print("  SIĞINAK HAVA KALİTESİ TAHMİN SİSTEMİ")
    print("=" * 60)
    print("Her zaman adımı için 4 değer girin:\n")
    print("  CO2      → Karbondioksit   (380 - 5000 ppm)")
    print("  Sıcaklık → Ortam ısısı     (16  - 34   °C)")
    print("  Nem      → Bağıl nem       (25  - 90   %)")
    print("  VOC      → Organik bileşik (50  - 1200 ppb)")
    print()

    t0 = manuel_giris("ZAMAN 1 — 25 saniye önce")
    print()
    t1 = manuel_giris("ZAMAN 2 — 20 saniye önce")
    print()
    t2 = manuel_giris("ZAMAN 3 — 15 saniye önce")
    print()
    t3 = manuel_giris("ZAMAN 4 — 10 saniye önce")
    print()
    t4 = manuel_giris("ZAMAN 5 — Şu an")

    sonuc = predictor.predict_shelter(t0, t1, t2, t3, t4)

    level = sonuc["alert"]["level"]
    emoji = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}.get(level, "⚪")

    print("\n" + "=" * 60)
    print("  TAHMİN SONUCU")
    print("=" * 60)
    print(f"  AQI Skoru : {sonuc['aqi_score']} / 100")
    print(f"  Durum     : {emoji} {sonuc['alert']['message']}")
    if sonuc["alert"]["action"]:
        print(f"  Eylem     : {sonuc['alert']['action']}")
    print(f"  Güven     : %{int(sonuc['confidence'] * 100)}")
    print(f"  Zaman     : {sonuc['timestamp']}")
    print("=" * 60)

    os.makedirs("output", exist_ok=True)
    with open("output/latest.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print("\n✅ Sonuç output/latest.json dosyasına kaydedildi.")