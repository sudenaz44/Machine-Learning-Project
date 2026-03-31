Vehicle Price Prediction - README.md
1️⃣ Proje Hakkında

Bu proje, ikinci el araç fiyatlarını tahmin etmek için oluşturulmuştur. Amaç: modelin araç özelliklerini kullanarak fiyatı tahmin etmesi.
Veri kaynağı: CSV formatında araç ilanları (US odaklı).

Modelleme için XGBoost ve LightGBM tercih edildi. Ayrıca feature engineering ve categorical encoding ile veriler model için hazırlandı.

2️⃣ Veri Temizleme Adımları
CSV Yükleme
import pandas as pd

df = pd.read_csv("cleaned_data.csv", engine="python", on_bad_lines='skip')
İlk İnceleme
.info() ve .head() ile veri tipleri, null değerler kontrol edildi.
Önemli kolonlar: price, engine_cylinders, km, car_age, state, brand, model, fuel, gearbox, drive_type, condition, title_status, type.
Eksik / Outlier Değerler
engine_cylinders ve car_age gibi sayısal değerlerde eksikler median ile dolduruldu.
Outlier’lar (fiyat = 0 veya aşırı km) temizlendi:
df = df[df['price'] > 0]
df = df[df['km'] < df['km'].quantile(0.99)]
Sayısal Dönüşümler
engine_cylinders string → numeric:
df['engine_cylinders'] = df['engine_cylinders'].astype(str).str.replace(' cylinders', '')
df['engine_cylinders'] = pd.to_numeric(df['engine_cylinders'], errors='coerce')
df['engine_cylinders'].fillna(df['engine_cylinders'].median(), inplace=True)
3️⃣ Feature Engineering & Encoding
Modelin anlayacağı format için kategorik kolonlar encode edildi:
Label Encoding: brand, model, state (çok fazla unique değer olduğu için)
One-Hot Encoding: fuel, gearbox, drive_type, condition, title_condition, car_type
from sklearn.preprocessing import LabelEncoder

le_brand = LabelEncoder()
df['brand_encoded'] = le_brand.fit_transform(df['brand'])

le_model = LabelEncoder()
df['model_encoded'] = le_model.fit_transform(df['model'])

le_state = LabelEncoder()
df['state_encoded'] = le_state.fit_transform(df['state'])

# One-hot encoding örneği:
df = pd.get_dummies(df, columns=['fuel', 'gearbox', 'drive_type', 'condition', 'title_condition', 'car_type'])
Orijinal string kolonlar model için kullanılmadı (city, brand, model, state) ama saklandı.
4️⃣ Model İçin Feature ve Target Seçimi
# Target
y = df['price']

# Modelin anlayacağı kolonlar
X = df.select_dtypes(exclude=['object'])
X = X.drop(columns=['price'])  # price target olduğu için çıkarıldı
5️⃣ Modelleme
XGBoost Regresyon
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

# Ölçümler
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

Sonuçlar örnek:

MAE: 2642
RMSE: 4620
R2: 0.86
Önemli Özellikler
import pandas as pd

fi = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
}).sort_values(by='importance', ascending=False)

print(fi.head(10))

Örnek:

feature	importance
car_age	0.34
km	0.27
drive_type_fwd	0.07
model_encoded	0.06
brand_encoded	0.03
6️⃣ Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1]
}

random_search = RandomizedSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

y_pred_best = best_model.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print("Tuned MAE:", mae_best)
print("Tuned RMSE:", rmse_best)
print("Tuned R2:", r2_best)
7️⃣ Görselleştirme Önerileri
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=fi.head(15))
plt.title("Top 15 Feature Importance")
plt.show()

# Predicted vs True
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, alpha=0.3)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs True Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
8️⃣ Kazanımlar & Öğrenilenler
Verilerin temizlenmesi, outlier’ların çıkarılması ve encoding ile modelin anlaması gereken format hazırlandı.
XGBoost ile MAE ~2.6k, RMSE ~4.6k ve R2 ~0.86 gibi yüksek başarı sağlandı.
Feature importance ile en etkili değişkenler tespit edildi: car_age, km, drive_type vs.
Hyperparameter tuning ile performans artırılabilir.