import pandas as pd
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

df = pd.concat([X, y], axis=1)

df.tail()
print(df.isnull().sum())
print(df['Bare_nuclei'])

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

grab_outliers(df, "Bare_nuclei")

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Bare_nuclei")

df["Bare_nuclei"]=df["Bare_nuclei"].fillna(df["Bare_nuclei"].mean())

X = df.drop("Class", axis=1)
y = df["Class"]

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

cv_results=cross_validate(mlpcl, X, y, cv=5, scoring=["accuracy","roc_auc"])
cv_results["test_accuracy"].mean()
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



cv_results=cross_validate(mlpcl, X, y, cv=5, scoring=["accuracy","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_roc_auc"].mean()
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

def encrypt_to_fixed_numeric(data, key, iv, fixed_length=16):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    data_bytes = data.astype(np.float32).tobytes()
    padded_data = data_bytes.ljust((len(data_bytes) + 15) // 16 * 16, b'\0')
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    print(encrypted_data)
    encrypted_array = np.frombuffer(encrypted_data, dtype=np.uint8)
    print(encrypted_array)
    if len(encrypted_array) < fixed_length:
        return np.pad(encrypted_array, (0, fixed_length - len(encrypted_array)), 'constant')
    else:
        return encrypted_array[:fixed_length]


key = os.urandom(32)
iv = os.urandom(16)


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_encrypted = df.copy()

# Encrypt numeric columns
for col in numeric_columns:
    df_encrypted[col] = df[col].apply(lambda x: encrypt_to_fixed_numeric(np.array([x]), key, iv).tolist())

# Label Encoding for the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Class'])  # Use the target 'Class' column for encoding

# Convert encrypted data (numeric columns) into a numpy array
X = np.array(df_encrypted[numeric_columns].values.tolist())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=0)

# Reshape the data if necessary
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the MLP model
mlpcl = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=17, solver='sgd')
mlpcl.fit(X_train, y_train)

# Make predictions
predictions = mlpcl.predict(X_test)

# Print the results
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


