# Sığınak Hava Kalitesi Erken Uyarı Sistemi

Afet sığınaklarındaki hava kalitesini izleyerek kritik düşüşleri 
önceden tespit eden makine öğrenmesi modeli.

## Proje Hakkında

Bu proje, KTO Karatay Üniversitesi Disiplinlerarası Proje dersi 
kapsamında geliştirilmiştir. Sığınak içerisindeki CO2, sıcaklık, 
nem ve VOC sensör verilerini analiz ederek havalandırma sistemi 
için erken uyarı üretmektedir.

## Kurulum

pip install pandas numpy scikit-learn joblib fastapi uvicorn

## Kullanım

python src/main.py

## Veri Seti

Kaggle: https://www.kaggle.com/datasets/emre26/shelter-air-quality-sensor-data

## Teknolojiler

- Python, scikit-learn, pandas, numpy
- Random Forest Classifier
- FastAPI

## Proje Yapısı

src/
├── data_loader.py   # Veri yükleme
├── preprocessor.py  # Ön işleme ve AQI hesaplama
├── trainer.py       # Model eğitimi
├── predictor.py     # Tahmin fonksiyonu
└── main.py          # Ana çalıştırma
