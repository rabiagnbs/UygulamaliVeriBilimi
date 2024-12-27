import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################

df=pd.read_csv('/Users/rabiagnbs/Desktop/VeriBilimi/pythonProject/diabets/diabetes.csv')
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################
#Bu problemin özelinde bir standartlaşma işlemi yapmamız gerekiyor.

y=df["Outcome"]
X=df.drop(["Outcome"], axis=1)

X_scaled=StandardScaler().fit_transform(X)#Burada X nump arrayi döner.
#Numpy arrayi döndüğü için değişkenleri standartlaştırmış olsa da elimizdeki array istediğimiz bilgileri içermediği için
#aşağıdaki gibi sütun isimleri ekleriz.

X=pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model= KNeighborsClassifier().fit(X, y)

random_user = X.sample(1,random_state=45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################
y_pred = knn_model.predict(X) #bütün gözlem birimleri için knn modelini kullanarak tahminde bulunuyoruz.
#y_pred hesaplama amacımız confusion matrix'tir.

#1 sınıfına ait olma olasılıklarını hesaplama (AUC için y_prob):
y_prob= knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))

#ROC AUC: Modelin pozitif ve negatif sınıfları ayırt etme yeteneğini ölçer.
roc_auc_score(y, y_prob)

#burada scoring eklememizin sebebi doğrulamamızı tek bir metriğe göre değil birden fazla metriğe bakarak yapacağız.
cv_results= cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "roc_auc","f1"])
cv_results['test_accuracy'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_f1'].mean()

#Başarı değerleri nasıl arttırılabilir?
# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()
#Parametre modelleri veri içinden öğrendiği ağırlıklardır.
#Hiper parametre ise kullanıcı tarafından tanımlanması gereken öğrenilemeyen parametrelerdir.

################################################
# 5. Hyperparameter Optimization
################################################


#Knn de öncelikli amacımız komşuluk sayısını değiştirmek.
knn_params ={"n_neighbors":range(2,50)}


#GridSearch ile her bir komşuluk knn modeli kurup hatamıza bakacağız.
#Eğer birden fazla parametre varsa her birinin kombinasyonlarını cross validation yaparak en iyi sonucu bulmaya çalışır GridSearchCv.

knn_gs_best=GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
#n_jobs=-1 işlemcileri tam performans yaparak kullanır.
#verbose=raporu beklemek demektir.

knn_gs_best.best_params_

################################################
# 6. Final Model
################################################

#en iyi değerini bulduysak bununla tekrar model kurmamız lazım.
#set_params ile biz atarız en iyiyi.
knn_final =knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results= cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "roc_auc","f1"])
cv_results['test_accuracy'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_f1'].mean()

random_user= X.sample(1, random_state=45)
knn_final.predict(random_user)

