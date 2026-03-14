# 🚗 Car Price ML Pipeline – End to End Project

## 📌 Proje Hakkında
Bu proje, Hindistan pazarındaki araçların özelliklerine göre **satış fiyatlarını tahmin eden bir makine öğrenmesi pipeline**'ını göstermektedir.  
Amaç, veri yükleme, keşifsel analiz, veri temizleme, özellik mühendisliği, model eğitimi ve değerlendirmeyi tek bir akışta sunmaktır.

---

## 📌 Kullanılan Veri
- Dataset: `cardekho.csv`  
- Satır sayısı: 8128 araç  
- Özellikler: km, motor gücü, yakıt türü, sahibin tipi, vites tipi, yaş, motor hacmi, vs.  

---

## 📌 Pipeline Aşamaları

1. **Data Loading & Inspection**  
   - CSV verisi yüklendi ve ilk inceleme yapıldı.

2. **Exploratory Data Analysis (EDA)**  
   - Fiyat dağılımı, km vs fiyat ilişkisi, istatistikler incelendi.

3. **Data Cleaning**  
   - Eksik veriler median ile dolduruldu.  
   - `max_power` sayısala çevrildi.  
   - Gereksiz sütunlar kaldırıldı.

4. **Feature Engineering**  
   - `car_age` gibi yeni özellikler üretildi.  
   - Kategorik değişkenler **One-Hot Encoding** ile sayısala çevrildi.

5. **Model Training & Evaluation**  
   - Modeller: Linear Regression, Decision Tree, Random Forest  
   - Performans:
     - **Random Forest**: MAE ≈ 70,481, R² = 0.97 → en iyi performans  
     - Decision Tree: R² = 0.94  
     - Linear Regression: R² = 0.70  

6. **Feature Importance**  
   - `max_power`, `car_age` ve `km_driven` en etkili özellikler olarak belirlendi.  
   - Bu grafik, modelin fiyatı tahmin ederken hangi özelliklere daha çok güvendiğini gösterir.

---

## 📌 Kullanılan Teknolojiler
- Python (pandas, matplotlib, seaborn)  
- Scikit-learn (Linear Regression, Decision Tree, Random Forest)  
- Google Colab (kod ve görselleştirme)  

---

## 📌 Sonuç
- Random Forest modeli ile araç fiyatlarını yüksek doğrulukla tahmin edebiliyoruz.  
- Bu proje, **ML pipeline’ını baştan sona gösteren bir portfolyo projesidir**.  
- GitHub’da paylaşmak ve CV’de göstermek için uygundur.

---

## 📌 İlerleyen Adımlar
- Model hiperparametre optimizasyonu (GridSearchCV, RandomizedSearchCV)  
- Daha gelişmiş modeller (XGBoost, LightGBM)  
- Web uygulaması ile fiyat tahmin sistemi  
