import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

hepatitis = fetch_ucirepo(id=46)

X = hepatitis.data.features
y = hepatitis.data.targets

df = pd.concat([X, y], axis=1)

df.tail()
print(df.isnull().sum())
df["Class"].nunique()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
df.columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["Steroid"]=df["Steroid"].fillna(df["Steroid"].mode()[0]).isnull().sum()
df["Fatigue"]=df["Fatigue"].fillna(df["Fatigue"].mode()[0]).isnull().sum()
df["Malaise"]=df["Malaise"].fillna(df["Malaise"].mode()[0]).isnull().sum()
df["Anorexia"]=df["Anorexia"].fillna(df["Anorexia"].mode()[0]).isnull().sum()
df["Liver Big"]=df["Liver Big"].fillna(df["Liver Big"].mode()[0]).isnull().sum()
df["Liver Firm"]=df["Liver Firm"].fillna(df["Liver Firm"].mode()[0]).isnull().sum()
df["Spleen Palpable"]=df["Spleen Palpable"].fillna(df["Spleen Palpable"].mode()[0]).isnull().sum()
df["Spiders"]=df["Spiders"].fillna(df["Spiders"].mode()[0]).isnull().sum()
df["Ascites"]=df["Ascites"].fillna(df["Ascites"].mode()[0]).isnull().sum()
df["Varices"]=df["Varices"].fillna(df["Varices"].mode()[0]).isnull().sum()
df["Spiders"]=df["Spiders"].fillna(df["Spiders"].mode()[0]).isnull().sum()
df["Bilirubin"]=df["Bilirubin"].fillna(df["Bilirubin"].mean()).isnull().sum()
df["Alk Phosphate"]=df["Alk Phosphate"].fillna(df["Alk Phosphate"].mean()).isnull().sum()
df["Sgot"]=df["Sgot"].fillna(df["Sgot"].mean()).isnull().sum()
df["Albumin"]=df["Albumin"].fillna(df["Albumin"].mean()).isnull().sum()
df["Protime"]=df["Protime"].fillna(df["Protime"].mean()).isnull().sum()
print(df.isnull().sum())

df["Class"].nunique()

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=0)
scaler= StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlpcl= MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=10000)


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

mlpcl.fit(X_train_res, y_train_res.values.ravel())
predictions =mlpcl.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

cv_results = cross_validate(
    mlpcl, X_train_res, y_train_res, cv=10,
    scoring={"accuracy": "accuracy", "f1": "f1_weighted", "roc_auc": "roc_auc_ovr"}
)

print("Accuracy:", cv_results["test_accuracy"].mean())
print("F1:", cv_results["test_f1"].mean())
print("ROC AUC:", cv_results["test_roc_auc"].mean())

mlpcl.get_params()

mlpcl_params= {
   'hidden_layer_sizes':[(10, 10, 10, 10), (100,100)] ,
   'activation': ['relu','logistic'],
   'learning_rate_init': [0.001, 0.05]
}

mlpcl_params= GridSearchCV(mlpcl, mlpcl_params, cv=5, n_jobs=1, verbose=True).fit(X,y)

mlpcl_params.best_params_

mlpcl.set_params(**mlpcl_params.best_params_).fit(X,y)
mlpcl.fit(X_train_res, y_train_res.values.ravel())
predictions =mlpcl.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

cv_results = cross_validate(
    mlpcl, X_train_res, y_train_res, cv=10,
    scoring={"accuracy": "accuracy", "f1": "f1_weighted", "roc_auc": "roc_auc_ovr"}
)

print("Accuracy:", cv_results["test_accuracy"].mean())
print("F1:", cv_results["test_f1"].mean())
print("ROC AUC:", cv_results["test_roc_auc"].mean())
