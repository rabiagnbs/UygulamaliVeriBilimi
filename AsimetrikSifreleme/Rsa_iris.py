import numpy as np
import pandas as pd
import seaborn as sns
import struct

from keras.src.utils.module_utils import tensorflow
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("/Users/rabiagnbs/Desktop/Iris.csv")
df.head()


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range,3)
    low_limit = round(quartile1 - 1.5 * interquantile_range,3)
    return low_limit, up_limit



for col in df.select_dtypes(include=['float64', 'int64']).columns:
    print(col, outlier_thresholds(df, col))



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.select_dtypes(include=['float64', 'int64']).columns:
    print(col, check_outlier(df, col))


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    outliers = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    if not outliers.empty:
        print(outliers)

grab_outliers(df, "SepalWidthCm")



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "SepalWidthCm")


le = LabelEncoder()
df['Species_Encoding'] = le.fit_transform(df['Species']) + 1
df.head()


X=df.iloc[:, 1:5]
X.head()

y=df["Species_Encoding"]
y.head()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=17)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlpcl= MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='logistic',max_iter=10000)
mlpcl.fit(X_train, y_train.values.ravel())
predictions =mlpcl.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))


comparison_df = pd.DataFrame({
    'Gerçek Sonuçlar': y_test.values,
    'Tahmin Edilen Sonuçlar': predictions
})


MSE = np.square(np.subtract(y_test, predictions)).mean()
RMSE = np.sqrt(MSE)
print(MSE, RMSE)


#RSA
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
public_key = private_key.public_key()


pem_private_key = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption()
)

pem_public_key = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)


def number_to_bytes(number):
    return struct.pack('>d', number)

def encrypt_message(public_key, number):
    byte_data = number_to_bytes(number)
    encrypted = public_key.encrypt(
        byte_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted

def decrypt_message(private_key, encrypted_data):
    # Şifrelenmiş veriyi özel anahtar ile çöz
    decrypted_data = private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    # Çözülen baytları double (float64) formatında sayıya dönüştür
    original_number = struct.unpack('>d', decrypted_data)[0]
    return original_number

df_sifreli = pd.DataFrame()
df_sifreli['encrypted_SepalLengthCm'] = df['SepalLengthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_SepalWidthCm'] = df['SepalWidthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_PetalLengthCm'] = df['PetalLengthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_PetalWidthCm'] = df['PetalWidthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_Species_Encoding'] = df['Species_Encoding'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli.head()

first_encrypted_value = df_sifreli['encrypted_Species_Encoding'].iloc[1]  # Hatalı sütun adı düzeltildi
length_first_encrypted_value = len(first_encrypted_value)
print(length_first_encrypted_value)

def convert_to_binary(encrypted_data):
    return ''.join(f'{byte:08b}' for byte in encrypted_data)

df_binary = pd.DataFrame()
df_binary['binary_SepalLengthCm'] = df_sifreli['encrypted_SepalLengthCm'].apply(convert_to_binary)
df_binary['binary_SepalWidthCm'] = df_sifreli['encrypted_SepalWidthCm'].apply(convert_to_binary)
df_binary['binary_PetalLengthCm'] = df_sifreli['encrypted_PetalLengthCm'].apply(convert_to_binary)
df_binary['binary_PetalWidthCm'] = df_sifreli['encrypted_PetalWidthCm'].apply(convert_to_binary)
df_binary['binary_Species_Encoding'] = df_sifreli['encrypted_Species_Encoding'].apply(convert_to_binary)
df_binary.head()


first_binary_value = df_binary['binary_Species_Encoding'].iloc[1]
length_of_first_value = len(first_binary_value)

print(length_of_first_value)

def visualize_binary_values(dataframe):
    for index, row in dataframe.iterrows():
        binary_values = {
            'SepalLengthCm': row['binary_SepalLengthCm'],
            'SepalWidthCm': row['binary_SepalWidthCm'],
            'PetalLengthCm': row['binary_PetalLengthCm'],
            'PetalWidthCm': row['binary_PetalWidthCm'],
            'SpeciesEncoding': row['binary_Species_Encoding']
        }
        for feature, binary_value in binary_values.items():
            bits = [int(bit) for bit in binary_value]
            grid_size = int(np.sqrt(len(bits)))
            reshaped_bits = np.array(bits[:grid_size * grid_size]).reshape(grid_size, grid_size)
            plt.figure(figsize=(5, 5))
            plt.imshow(reshaped_bits, cmap='binary')
            plt.title(f'Pixel Grid of {feature} - Row {index}')
            plt.tight_layout()
            plt.show()

visualize_binary_values(df_binary)

