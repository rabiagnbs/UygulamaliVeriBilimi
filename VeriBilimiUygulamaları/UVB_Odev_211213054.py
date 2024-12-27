import pickle
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBModel, XGBClassifier

df = pd.read_csv("/Users/rabiagnbs/Desktop/train_odev.csv")
df.head()
def check_df(dataframe, head=5):
    print("##################### Satır ve Sütun Sayıları #####################")
    print(dataframe.shape)
    print("##################### Nitelik Tipleri #####################")
    print(dataframe.dtypes)
    print("##################### İlk 5 Değer #####################")
    print(dataframe.head(head))
    print("##################### Son 5 Değer #####################")
    print(dataframe.tail(head))
    print("##################### Eksik Değerler #####################")
    print(dataframe.isnull().sum())
    print("##################### Aykırı Değerler #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Ortalama Değerler #####################")
    print(df.mean())
    print("##################### Medyan Değerler #####################")
    print(df.median())
    print("##################### Mod Değerler #####################")
    print(df.mode().iloc[0])
    print("##################### Standart Sapma #####################")
    print(df.std())
    print("##################### Varyans #####################")
    print(df.var())
    print("##################### Beş Sayı Özeti #####################")
    print(df.describe().T)


    df.hist(bins=20, figsize=(20, 18))
    plt.suptitle("Histogramlar")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, width=0.5)
    plt.title("Kutu Grafik (Boxplot)")
    plt.xticks(rotation=90)
    plt.show()

    if 'class' in df.columns:
        sns.pairplot(df, hue='class')
        plt.suptitle("Dağılım Grafiği (Scatter Plot)")
        plt.show()

check_df(df)


correlation_matrix = df.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Korelasyon Matrisi", fontsize=20)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='price_range', data=df)
plt.title('Sınıf Dağılımı')
plt.xlabel('Price Range')
plt.ylabel('Frekans')
plt.show()


df.isnull().sum()


def preprocessing(df, q1=0.25, q3=0.75):

    def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    def drop_high_corr_features(dataframe, threshold=0.6):
        corr_matrix = dataframe.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        print(f"Kaldırılan Sütunlar: {to_drop}")
        return dataframe.drop(columns=to_drop)

    for col in df.select_dtypes(include=['int64', 'float64']).columns:  # Yalnızca sayısal sütunları işler
        if check_outlier(df, col):
            print(f"Aykırı değerler '{col}' sütununda bulundu. Eşik değerlerle değiştiriliyor...")
            replace_with_thresholds(df, col)
    drop_high_corr_features(df)
    return df

preprocessing(df)

df.head()

X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=17)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

catboost_model = CatBoostClassifier(random_state=17, verbose=False)
catboost_model.fit(X_train_res, y_train_res)
y_pred = catboost_model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
cv_results = cross_validate(
    catboost_model, X, y, cv=5,
    scoring={"accuracy": "accuracy", "f1": "f1_weighted", "roc_auc": "roc_auc_ovr"}
)

print("Accuracy:", cv_results["test_accuracy"].mean())
print("F1:", cv_results["test_f1"].mean())
print("ROC AUC:", cv_results["test_roc_auc"].mean())

catboost_params = {"iterations": [200, 500,700],
                   "learning_rate": [0.01, 0.05],
                   "depth": [3,5, 7]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_best_grid.best_estimator_
catboost_final.fit(X_train_res, y_train_res)
y_pred = catboost_final.predict(X_test)

print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
cv_results = cross_validate(
    catboost_final, X, y, cv=5,
    scoring={"accuracy": "accuracy", "f1": "f1_weighted", "roc_auc": "roc_auc_ovr"}
)
print("Accuracy:", cv_results["test_accuracy"].mean())
print("F1:", cv_results["test_f1"].mean())
print("ROC AUC:", cv_results["test_roc_auc"].mean())

with open('UVB_Odev_211213054.pkl', 'wb') as file:
    pickle.dump(catboost_final, file)

with open('UVB_Odev_211213054.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
