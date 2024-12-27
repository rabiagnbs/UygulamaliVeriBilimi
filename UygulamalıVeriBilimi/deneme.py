import pickle

# Modeli yükle
with open("/Users/rabiagnbs/Desktop/VeriBilimi/pythonProject/UVB_Odev_211213054.pkl", "rb") as file:
    loaded_model = pickle.load(file)

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Veri setini yükleme
df = pd.read_csv("/Users/rabiagnbs/Desktop/train_odev.csv")
    # Özellik ve hedef değişkenlerini ayırma
X_test = df.drop("price_range", axis=1)
y_test = df["price_range"]

    # Standartlaştırma
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
from sklearn.metrics import accuracy_score, classification_report

# Tahmin yap
y_pred = loaded_model.predict(X_test_scaled)

# Performansı değerlendirme
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


import joblib

# Model dosyasının yolu
file_path = "/Users/rabiagnbs/Desktop/VeriBilimi/pythonProject/UVB_Odev_211213054.pkl"

# Modeli yükle
model = joblib.load(file_path)

# scikit-learn sürümünü kontrol et
if isinstance(model, dict) and "scikit_learn_version" in model:
    print("Modelin scikit-learn sürümü:", model["scikit_learn_version"])
else:
    print("scikit-learn sürümü dosyadan okunamadı.")
