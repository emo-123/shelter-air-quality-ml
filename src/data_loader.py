# ─────────────────────────────────────────────
# VERİ YÜKLEME

# Sütunlar: timestamp, co2_ppm, temp_c, humidity, voc_ppb
# ─────────────────────────────────────────────

import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        print("📊 Veri seti boyutu:", df.shape)
        print("\nEksik değer var kontrolü")
        print(df.isnull().sum())
        return df