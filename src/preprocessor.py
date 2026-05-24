# ─────────────────────────────────────────────
# VERİ TEMİZLEME, AQI SKORU VE ETİKETLEME
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd

class Preprocessor:
    def clean(self, df):
        df.ffill(inplace=True) 

        df["co2_ppm"]  = df["co2_ppm"].clip(380, 5000)
        df["temp_c"]   = df["temp_c"].clip(16, 34)
        df["humidity"] = df["humidity"].clip(25, 90)
        df["voc_ppb"]  = df["voc_ppb"].clip(50, 1200)

        df.to_csv("data/shelter_sensor_dataset_v2_clean.csv", index=False)
        print("\nVeri temizlendi!")
        print("   Satır sayısı :", len(df))
        print("   Sütun sayısı :", len(df.columns))
        print("   Eksik değer  :", df.isnull().sum().sum())
        return df

    def air_quality_score(self, co2, voc, humid):
        co2_score   = np.clip(100 - (co2 - 400) / 46, 0, 100)
        voc_score   = np.clip(100 - (voc - 50) / 9.5, 0, 100)
        humid_score = np.clip(100 - abs(humid - 50) * 1.8, 0, 100)
        return co2_score * 0.5 + voc_score * 0.3 + humid_score * 0.2

    def add_aqi(self, df):
        df["aqi_score"] = self.air_quality_score(
            df["co2_ppm"], df["voc_ppb"], df["humidity"]
        )
        print("\nAQI skoru hesaplandı!")
        print(f"   En düşük : {df['aqi_score'].min():.2f}")
        print(f"   En yüksek: {df['aqi_score'].max():.2f}")
        print(f"   Ortalama : {df['aqi_score'].mean():.2f}")
        return df

    def build_model_dataset(self, df):
        records = []

        for i in range(4, len(df) - 1):
            t0 = df.iloc[i - 4]  # 25sn önce
            t1 = df.iloc[i - 3]  # 20sn önce
            t2 = df.iloc[i - 2]  # 15sn önce
            t3 = df.iloc[i - 1]  # 10sn önce
            t4 = df.iloc[i]      # şu an
            target_aqi = df.iloc[i + 1]["aqi_score"]

            if target_aqi >= 90:
                label = 0  # Normal
            elif target_aqi >= 70:
                label = 1  # Dikkat
            else:
                label = 2  # Kritik

            records.append({
                "t0_co2": t0["co2_ppm"], "t0_temp": t0["temp_c"],
                "t0_humid": t0["humidity"], "t0_voc": t0["voc_ppb"],
                "t1_co2": t1["co2_ppm"], "t1_temp": t1["temp_c"],
                "t1_humid": t1["humidity"], "t1_voc": t1["voc_ppb"],
                "t2_co2": t2["co2_ppm"], "t2_temp": t2["temp_c"],
                "t2_humid": t2["humidity"], "t2_voc": t2["voc_ppb"],
                "t3_co2": t3["co2_ppm"], "t3_temp": t3["temp_c"],
                "t3_humid": t3["humidity"], "t3_voc": t3["voc_ppb"],
                "t4_co2": t4["co2_ppm"], "t4_temp": t4["temp_c"],
                "t4_humid": t4["humidity"], "t4_voc": t4["voc_ppb"],
                "co2_trend":   t4["co2_ppm"]  - t0["co2_ppm"],
                "voc_trend":   t4["voc_ppb"]  - t0["voc_ppb"],
                "temp_trend":  t4["temp_c"]   - t0["temp_c"],
                "humid_trend": t4["humidity"] - t0["humidity"],
                "label": label
            })

        model_df = pd.DataFrame(records)
        print("\n Model veri seti oluşturuldu!")
        print(f"   Satır sayısı : {len(model_df)}")
        print(f"   Sütun sayısı : {len(model_df.columns)}")
        print("\nUyarı seviyesi dağılımı:")
        dagilim = model_df["label"].value_counts().sort_index()
        for seviye, isim in zip([0, 1, 2], ["Normal", "Dikkat", "Kritik"]):
            sayi = dagilim.get(seviye, 0)
            oran = sayi / len(model_df) * 100
            print(f"   {isim:8}: {sayi:6} satır ({oran:.1f}%)")
        return model_df