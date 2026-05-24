# ─────────────────────────────────────────────
# MODEL EĞİTİMİ
#
# Algoritma: Random Forest
#   - Yorumlanabilir ve hızlı
#   - Dengesiz veri için class_weight="balanced"
#   - Zaman serisi olduğu için shuffle=False
#
# Eğitim/Test: %80 / %20
# ─────────────────────────────────────────────

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class Trainer:
    def __init__(self):
        self.model = None

    def train(self, model_df):
        features = [c for c in model_df.columns if c != "label"]
        X = model_df[features]
        y = model_df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        print(f"📚 Eğitim seti : {X_train.shape}")
        print(f"🧪 Test seti   : {X_test.shape}")

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        print("\n✅ Model eğitimi tamamlandı!")

        y_pred = self.model.predict(X_test)
        print("\n📋 Sınıflandırma Raporu:")
        print(classification_report(
            y_test, y_pred,
            target_names=["Normal", "Dikkat", "Kritik"]
        ))

        joblib.dump(self.model, "shelter_model_v2.pkl")
        print("💾 Model kaydedildi: shelter_model_v2.pkl")

        return self.model, X_train

    def plot_feature_importance(self, model, X_train):
        importances   = model.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 7))
        plt.barh(
            [feature_names[i] for i in indices],
            [importances[i] for i in indices],
            color="#378ADD"
        )
        plt.xlabel("Önem Skoru")
        plt.title("Hangi Özellik Ne Kadar Önemli?")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=120)
        plt.show()
        print("✅ Grafik kaydedildi: feature_importance.png")