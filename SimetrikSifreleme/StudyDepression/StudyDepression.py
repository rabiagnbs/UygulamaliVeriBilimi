import kagglehub
import pandas as pd
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Veri yükleme ve işleme
path = kagglehub.dataset_download("ikynahidwin/depression-student-dataset")

files = os.listdir(path)
print("Files in the dataset directory:", files)

# CSV veya Excel dosyasını seç
for file in files:
    if file.endswith(".csv"):  # CSV dosyası
        data_file = os.path.join(path, file)
        df = pd.read_csv(data_file)
        print(df.head())
        break
    elif file.endswith(".xlsx"):  # Excel dosyası
        data_file = os.path.join(path, file)
        df = pd.read_excel(data_file)
        print(df.head())
        break
else:
    print("No readable file found in the dataset directory.")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)

# Download the dataset
path = kagglehub.dataset_download("ikynahidwin/depression-student-dataset")

# Print the path to the dataset
print("Path to dataset files:", path)

# Dataset'in içindeki dosyaları listele
files = os.listdir(path)
print("Files in the dataset directory:", files)

# CSV veya Excel dosyasını seç
for file in files:
    if file.endswith(".csv"):  # CSV dosyası
        data_file = os.path.join(path, file)
        df = pd.read_csv(data_file)
        print(df.head())
        break
    elif file.endswith(".xlsx"):  # Excel dosyası
        data_file = os.path.join(path, file)
        df = pd.read_excel(data_file)
        print(df.head())
        break
else:
    print("No readable file found in the dataset directory.")

df['Dietary Habits'].nunique()


le = LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
df["Dietary Habits"]=le.fit_transform(df["Dietary Habits"])
df["Sleep Duration"]=le.fit_transform(df["Sleep Duration"])
df["Family History of Mental Illness"]=le.fit_transform(df["Family History of Mental Illness"])
df["Depression"]=le.fit_transform(df["Depression"])
df["Have you ever had suicidal thoughts ?"]=le.fit_transform(df["Have you ever had suicidal thoughts ?"])

df.head()


X = df.drop("Depression", axis=1)
y = df["Depression"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=0)
scaler= StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlpcl= MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=10000)
mlpcl.fit(X_train, y_train.values.ravel())
predictions =mlpcl.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

cv_results=cross_validate(mlpcl, X, y, cv=5, scoring=["accuracy", "f1" ,"roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

mlpcl.get_params()

mlpcl_params= {
   'hidden_layer_sizes':[(10, 10, 10, 10), (100,100)] ,
   'activation': ['relu','logistic'],
   'learning_rate_init': [0.001, 0.05]
}

mlpcl_params= GridSearchCV(mlpcl, mlpcl_params, cv=5, n_jobs=1, verbose=True).fit(X,y)

mlpcl_params.best_params_

mlpcl.set_params(**mlpcl_params.best_params_).fit(X,y)

cv_results=cross_validate(mlpcl, X, y, cv=5, scoring=["accuracy", "f1" ,"roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
# AES şifreleme fonksiyonu


# AES şifreleme fonksiyonu
def encrypt_to_aes(data, key, iv):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Veriyi byte formatına dönüştür
    data_bytes = data.astype(np.float32).tobytes()

    # Veriyi 16 baytlık bloklara göre padding yap
    padded_data = data_bytes.ljust((len(data_bytes) + 15) // 16 * 16, b'\0')

    # Veriyi şifrele
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    return encrypted_data


# Sayısal formata dönüştürme fonksiyonu
def convert_encrypted_to_numeric(encrypted_data, fixed_length=16):
    # Şifreli veriyi sayısal formata dönüştür
    encrypted_numeric = np.frombuffer(encrypted_data, dtype=np.uint8)

    # Eğer şifreli veri sabit uzunluktan kısa ise, sıfırlarla padding yapılır
    if len(encrypted_numeric) < fixed_length:
        return np.pad(encrypted_numeric, (0, fixed_length - len(encrypted_numeric)), 'constant')
    else:
        return encrypted_numeric[:fixed_length]


# Şifreleme işlemi
def encrypt_dataframe(df, key, iv):
    """
    DataFrame içindeki tüm sayısal verileri AES ile şifreler.
    """
    encrypted_df = df.copy()

    # Sayısal veri türündeki her hücreyi şifrele
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        encrypted_df[column] = df[column].apply(
            lambda x: encrypt_to_aes(np.array([x]), key, iv)
        )

    return encrypted_df


# Sayısal formata dönüştürme işlemi
def convert_to_numeric_format(df, fixed_length=16):
    """
    Şifreli veri çerçevesini sayısal formata dönüştürür.
    """
    numeric_df = df.copy()

    # Şifrelenmiş veri içeren her hücreyi sayısal formata dönüştür
    for column in df.columns:
        numeric_df[column] = df[column].apply(
            lambda x: convert_encrypted_to_numeric(x, fixed_length)
        )

    return numeric_df


# AES şifreleme için anahtar ve IV oluşturma
key = os.urandom(32)  # AES-256 için 32 baytlık anahtar
iv = os.urandom(16)  # AES için 16 baytlık IV


# 1. Veri çerçevesini şifrele
df_encrypted = encrypt_dataframe(df, key, iv)
print("Şifrelenmiş Veri Çerçevesi:")
print(df_encrypted)

# 2. Şifrelenmiş veriyi sayısal formata dönüştür
df_numeric = convert_to_numeric_format(df_encrypted, fixed_length=16)
print("\nSayısal Formata Dönüştürülmüş Veri Çerçevesi:")
print(df_numeric["Gender"].head())


# X ve y'yi ayırma
X = np.array(df.drop(columns=["Depression"]).apply(lambda col: col.tolist() if isinstance(col.iloc[0], list) else col, axis=0).values.tolist())
y = np.array(df["Depression"].tolist())

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model oluşturma ve eğitme
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Modelin doğruluğunu değerlendirme
predictions = mlp.predict(X_test)

# Sonuçları yazdırma
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

cv_results=cross_validate(mlpcl, X, y, cv=5, scoring=["accuracy", "f1" ,"roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()