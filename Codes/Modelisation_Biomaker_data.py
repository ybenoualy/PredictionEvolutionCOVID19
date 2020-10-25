# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Modélisation 
# MAGIC 
# MAGIC Le projet choisit est un sujet de type de la santé en rapport avec la crise sanitaire COVID 19 . Il s'agit d'un apprentissage de type suppervisé d'une problématique de classification de patients présentant de symptomes simmilaires au COVID 19.
# MAGIC 
# MAGIC Nous mettons en place la variable target : Patient décédé ou non suite à la contraction du virus. 
# MAGIC ## Sommaire
# MAGIC 
# MAGIC ### I.	Gathering data 
# MAGIC 
# MAGIC 1. Open data Health 
# MAGIC 
# MAGIC 2. Problématique et identification de données 
# MAGIC 
# MAGIC 
# MAGIC ### II. ANALYSE EXPLORATOIRE DES DONNEES
# MAGIC 1.	Echantillonnage
# MAGIC 
# MAGIC 2.	Features engineering 
# MAGIC 
# MAGIC 3.	Visualisation des données
# MAGIC 
# MAGIC ### III.	PREPARATION DES DONNEES 
# MAGIC 
# MAGIC 1.	Répartition de données en features X et Target Y
# MAGIC 
# MAGIC     1.1 Features 
# MAGIC     
# MAGIC     1.2 Target 
# MAGIC     
# MAGIC     
# MAGIC 2.	Selection des variables
# MAGIC 
# MAGIC ### IV.	MODELISATION 
# MAGIC 
# MAGIC 1. Description du process de modélisation
# MAGIC 2.	Hyperparameters tuning
# MAGIC 
# MAGIC 3.	Evaluation 
# MAGIC 
# MAGIC ### V.	CONCLUSION

# COMMAND ----------

# MAGIC %sh
# MAGIC #pip install matplotlib
# MAGIC #pip install numpy
# MAGIC #pip install pandas==0.25.1
# MAGIC ###conda install -c anaconda basemap
# MAGIC pip install scikit-learn
# MAGIC pip install spark_df_profiling
# MAGIC pip install pydotplus
# MAGIC 
# MAGIC pip install missingno
# MAGIC pip install graphviz
# MAGIC pip install xgboost 
# MAGIC pip install imblearn
# MAGIC pip install lightgbm
# MAGIC pip install --upgrade scikit-learn

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Importation de librairies 

# COMMAND ----------

# Import libraries

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import pydotplus
from re import sub
import scipy.stats as sci
import seaborn as sns
import spark_df_profiling
from pathlib import Path


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import SVG
from graphviz import Source
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

#imports
#basics
import pandas as pd
import numpy as np
import random
from math import sqrt
from scipy.stats import gaussian_kde

#sklearn tools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
#classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
#regressions
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
#plotting
import matplotlib.pyplot as plt
import seaborn as sns

#set options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
 
# Set visualization prefrences 
sns.set(font_scale=1.5, style="darkgrid")
pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## 1. Gathering data 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Open data Health 
# MAGIC 
# MAGIC Plusieurs open data Heath de la santé ont été rendus publiques pendant la crise sanitaire pour permettre aux chercheurs de comprendre la propahgation de la maladie et aussi faciliter la recherche médicale du vaccin. 
# MAGIC 
# MAGIC Voici une lise non exhautive de quelques open data : 
# MAGIC 
# MAGIC - https://covid19researchdatabase.org 
# MAGIC - https://datasetsearch.research.google.com 
# MAGIC - https://www.kaggle.com/imdevskp/corona-virus-report
# MAGIC - https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
# MAGIC - https://www.kaggle.com/c/stanford-covid-vaccine
# MAGIC - https://www.worldometers.info/coronavirus/ 
# MAGIC 
# MAGIC - https://coronavirus.jhu.edu/us-map
# MAGIC - https://www.ecdc.europa.eu/en/covid-19/data 
# MAGIC - https://healthdata.gov 
# MAGIC - https://www.re3data.org
# MAGIC - http://www.chdstudies.org/research/information_for_researchers.php
# MAGIC - http://leo.ugr.es/elvira/DBCRepository/
# MAGIC - https://seer.cancer.gov/explorer/
# MAGIC - http://www.oasis-brains.org
# MAGIC - https://www.reddit.com/r/datasets/

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Nous avons constitué un dataset à partir de ces données
# MAGIC - https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/ 
# MAGIC - https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/subset_international_cases_2020_03_11.csv
# MAGIC - https://brunods10.s3-us-west-2.amazonaws.com/MIT_COVID/latestdata.csv

# COMMAND ----------



#Importing "merged" file from table. This file contains the input data for the program.

# File location and type
file_location = "/FileStore/tables/df_dummy2.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

## import merged file 
# The applied options are for CSV files. For other file types, these will be ignored.
df_dummy = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)




# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Problématique et identification de données
# MAGIC 
# MAGIC -  L'identifier les données de santé permettant de caractériser un patient atteint du COVID 19. 
# MAGIC - 	L’implémentation de ces algorithmes de machine Learning capables d’apprendre à partir de données et de déterminer la probabilité du décès du patient pour pouvoir identifier les patients fragiles et éviter les morts par négligeance.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ### II. ANALYSE EXPLORATOIRE DES DONNEES

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Echantillonnage 

# COMMAND ----------

## transformer pyspark dataframe à pandas dataframe 
pandas_df_dummy = df_dummy.toPandas()


# COMMAND ----------

pandas_df_dummy.info()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Nous avons les variables du dataset contenant les symptômes des patients, les caractérisques socio, le pays ... 
# MAGIC 
# MAGIC Parmi les variables : 
# MAGIC 
# MAGIC - pneumonia: une infection aiguë des voies aériennes inférieures caractérisée par une atteinte inflammatoire, voire purulente, du parenchyme pulmonaire (bronchioles, alvéoles pulmonaires et interstitium pulmonaire).
# MAGIC 
# MAGIC - rhinorrhea :  un écoulement nasal. Ce terme médical désigne la morve. La rhinorrhée peut survenir à la suite d'un traumatisme, d'un os comportant un ou plusieurs sinus aériens en communication avec les fosses nasales. On a par exemple l'éthmoïde, le sphénoïde,   l'os frontal entre autres.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC -  ILI : Une infection respiratoire aiguë avec: fièvre mesurée ≥ 38 C°, toux. 
# MAGIC 
# MAGIC - ARDS : Syndrome de détresse respiratoire aigu : Le syndrome de détresse respiratoire aigu (ARDS) est un état potentiellement fatal où les poumons ne peuvent pas fournir assez d'oxygène aux  organes vitaux du fuselage.

# COMMAND ----------

pandas_df_dummy.head()

# COMMAND ----------

#Define missing data function to identify the total number of missing data and associated percentage 
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

# COMMAND ----------

pandas_df_dummy.shape

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Nous avons 3912 lignes et 50 colonnes(50 variables)

# COMMAND ----------

missing_data(pandas_df_dummy)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Identifier les variables présentant un pourcentage élévé de données manquantes 
# MAGIC 
# MAGIC Par exemple, pour la variable : outcome2 3632 lignes manquantes et un pourcentage 92%. 

# COMMAND ----------

## Identifier les valeurs uniques de la variable outcome2 
pandas_df_dummy['outcome2'].unique()

# COMMAND ----------

## Identifier les valeurs uniques de la variable dead

pandas_df_dummy['dead'].unique()

# COMMAND ----------

## Identifier les valeurs uniques de la variable outcome

pandas_df_dummy['outcome'].unique()

# COMMAND ----------

## Identifier les valeurs uniques de la variable Death 
pandas_df_dummy['Death'].unique()

# COMMAND ----------

## Identifier les valeurs uniques de la variable sex

pandas_df_dummy['sex'].unique()

# COMMAND ----------

## 
## Identifier les valeurs uniques de la variable Admission
pandas_df_dummy['Admission'].unique()


# COMMAND ----------

# Supprimer les lignes  de valeurs  NAN 
pandas_df_dummy.dropna(how = 'all', inplace = True)

# COMMAND ----------


# Dataframe information
pandas_df_dummy.info()

# COMMAND ----------

patient_features = ['sex','age','symptoms', 'date_onset_symptoms', 'date_admission_hospital', 'date_death_or_discharge', 'date_confirmation']

geography_features = ['country', 'continent']



Features_Symptoms = ['cough','fever','dyspnea', 'other_respiratory', 'fatigue', 'myalgia', 'other_neurological', 'emesis', 'other_systemic', 'other_gastrointestinal', 'asthenia', 'other_muscoloskeletal', 'pneumonia', 'rhinorrhea', 'pharyngitis', 'other_musculoskeletal', 'other_ocular', 'nausea', 'malaise', 'ILI', 'pain', 'ARDS', 'sepsis', 'other_cardiovascular', 'AKD', 'organ.failure']

structured_features = ['days_to_event', 'dead', 'hospitalized', 'outcome2', 'Admission', 'Recovery', 'Death', 'time_til_admission', 'time_til_recovery', 'time_til_death']

probale_target = ['dead', 'outcome', 'outcome2', 'Death']

# COMMAND ----------

### Identifier les types de chaque variable 

pandas_df_dummy.dtypes

# COMMAND ----------

# Comparaison de la proportion entre la variable Death et outcome2 
print('NAs in Death: ', pandas_df_dummy['Death'].isna().sum())
print('NAs in outcome2: ', pandas_df_dummy['outcome2'].isna().sum())

print('% different Death: ', sum(pandas_df_dummy['Death']!=pandas_df_dummy['outcome2'])/pandas_df_dummy.shape[0]*100)



# COMMAND ----------

#Vérifier le pourcentage maximal   NA restantes  
round(np.max(pandas_df_dummy.isna().sum())/pandas_df_dummy.shape[0]*100,2)

# COMMAND ----------

for feature in Features_Symptoms:
  pandas_df_dummy[feature] = pandas_df_dummy[feature]*1000/pandas_df_dummy['pneumonia']

# COMMAND ----------

# Verifier toutes les variables et rassurer qu'elles soient numériques.
pandas_df_dummy[pandas_df_dummy.columns[~pandas_df_dummy.columns.isin(probale_target)]].info()

# COMMAND ----------

pandas_df_dummy.head()

# COMMAND ----------

## Remplacer NA par NaN 
pandas_df_dummy['time_til_recovery'] = pandas_df_dummy['time_til_recovery'].replace(['NA'],'NaN')
pandas_df_dummy['time_til_death'] = pandas_df_dummy['time_til_death'].replace(['NA'],'NaN')
pandas_df_dummy['time_til_admission'] = pandas_df_dummy['time_til_admission'].replace(['NA'],'NaN')
pandas_df_dummy['age'] = pandas_df_dummy['age'].replace(['NA'],'NaN')
pandas_df_dummy['sex'] = pandas_df_dummy['sex'].replace(['NA'],'NaN')

# COMMAND ----------

### Faire un group By 
pandas_df_dummy_outcome2_Death = pandas_df_dummy.groupby('outcome2')['Death'].sum().reset_index()

# COMMAND ----------


## Supprimer les variales non censées 
pandas_df_dummy = pandas_df_dummy.drop(['_c0', 'symptoms', 'date_onset_symptoms', 'date_admission_hospital', 'date_death_or_discharge', 'date_confirmation', 'continent', 'country'], axis=1)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Analyse de la variable target 
# MAGIC 
# MAGIC Variable Target Death : La variable prend 0 pour non décès du patient et 1 pour les patients décédés

# COMMAND ----------

pandas_df_dummy['Death'].unique()

# COMMAND ----------

## Répartition de la variable target Death (décès)
pandas_df_dummy['Death'].value_counts().plot.barh()

# COMMAND ----------

## Regrouper les variables par rapport à la target 

percent_target = pandas_df_dummy.groupby('Death').count()
percent_target['percent'] = 100*(percent_target['age']/pandas_df_dummy['Death'].count())
percent_target.reset_index(level=0, inplace=True)
percent_target

# COMMAND ----------

import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Patient décedé','Patient non décédé'
sizes = [2, 98]
explode = (0.1, 0)  # only "explode" the 1st slice (i.e. 'Toxic contents')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



# COMMAND ----------

# MAGIC %md 
# MAGIC Nous avons 98% de patients non décédés et 2 % de patients décédés dans notre échantillon 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Features engineering 

# COMMAND ----------

# les valeurs manquantes 
msno.bar(pandas_df_dummy, figsize=(16, 4))

# COMMAND ----------

## Remplacer les valeurs manquantes par la moyenne 

pandas_df_dummy = pandas_df_dummy.fillna(pandas_df_dummy.mean())

# COMMAND ----------

(pandas_df_dummy.isnull().sum() == pandas_df_dummy.shape[0]).any()

# COMMAND ----------

msno.bar(pandas_df_dummy, figsize=(16, 4))

# COMMAND ----------

pandas_df_dummy.head()

# COMMAND ----------

(pandas_df_dummy.isnull().sum() == pandas_df_dummy.shape[0]).any()

# COMMAND ----------

full_null_data = (pandas_df_dummy.isnull().sum() == pandas_df_dummy.shape[0])
full_null_columns = full_null_data[full_null_data == True].index

# COMMAND ----------

## colonnes avec toutes les valeurs égales à 0  

print(full_null_columns.tolist())

# COMMAND ----------

pandas_df_dummy.drop(full_null_columns, axis=1, inplace=True)

# COMMAND ----------

(pandas_df_dummy.isnull().sum() / pandas_df_dummy.shape[0]).sort_values(ascending=False).head()

# COMMAND ----------

contain_null_series = (pandas_df_dummy.isnull().sum() != 0).index

# COMMAND ----------

not_null_series = (pandas_df_dummy.isnull().sum() == 0)
not_null_columns = not_null_series[not_null_series == True].index
not_null_columns = not_null_columns[1:]

# COMMAND ----------

#drop_cols = ['id','_c0', 'dead', 'outcome2', 'outcome', 'time_til_recovery', 'time_til_death']
df = pandas_df_dummy 
pandas_df_dummy = pandas_df_dummy.drop(['dead', 'outcome2', 'outcome', 'time_til_recovery', 'time_til_death'], axis=1)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Visualisation des données

# COMMAND ----------

cols = pandas_df_dummy.columns.drop('id')

pandas_df_dummy[cols] = pandas_df_dummy[cols].apply(pd.to_numeric, errors='coerce')

# COMMAND ----------

sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 1]['ARDS'], label="Patient décédé")
sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 0]['ARDS'], label="Patient non décédé")
plt.legend()

# COMMAND ----------

sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 1]['pneumonia'], label="Patient décédé")
sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 0]['pneumonia'], label="Patient non décédé")
plt.legend()

# COMMAND ----------

sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 1]['malaise'], label="Patient décédé")
sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 0]['malaise'], label="Patient non décédé")
plt.legend()

# COMMAND ----------


sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 1]['other_muscoloskeletal'], label="Patient décédé")
sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 0]['other_muscoloskeletal'], label="Patient non décédé")
plt.legend()

# COMMAND ----------


sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 1]['days_to_event'], label="Patient décédé")
sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 0]['days_to_event'], label="Patient non décédé")
plt.legend()

# COMMAND ----------


sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 1]['age'], label="Patient décédé")
sns.distplot(pandas_df_dummy[pandas_df_dummy['Death'] == 0]['age'], label="Patient non décédé")
plt.legend()

# COMMAND ----------

list_var=['age', 'days_to_event', 'other_muscoloskeletal', 
          'malaise', 'pneumonia', 'ARDS']
def var_hist_global(df,X='Death',Y=list_var, Title='Features Engineering - Histograms', KDE=False):
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2 ,figsize=(14,16))#, sharey=True )
    aX = [ax1, ax2,ax3,ax4,ax5,ax6]
    for i in range(len(list_var)):   
        sns.distplot( df[list_var[i]][df[X]== 1 ].dropna(), label="Patient décédé" , ax= aX[i], kde= KDE , color = 'red')           
        sns.distplot( df[list_var[i]][df[X]== 0 ].dropna(), label="Patient non décédé"   , ax= aX[i], kde= KDE , color = "olive")
    plt.legend()
    plt.title(Title)
    plt.show()
    #plt.savefig("Features_Engineering_Histograms")
    
var_hist_global(df=pandas_df_dummy,X='Death',Y=list_var, Title='Histogramme Quora Questions', KDE=True)

# COMMAND ----------

# Calculate number of obs per group & median to position labels
list_var = ['age', 'ARDS', 'pneumonia']
def violin_boxplott(df,X='Death',Y=list_var, Title='Features Engineering - Box plot'): 
    fig, (ax1, ax2 ,ax3) = plt.subplots(1,3 ,figsize=(14,8))#, sharey=True )
    medians = pandas_df_dummy.groupby(['Death'])['age', 'ARDS', 'pneumonia'].median().values
 
    sns.boxplot( y=list_var[0],  x=X , data = df, ax= ax1 , palette=['olive','red'])
    sns.boxplot( y=list_var[1],  x=X , data = df, ax= ax2 , palette=['olive','red'])
    sns.boxplot( y=list_var[2],  x=X , data = df, ax= ax3 , palette=['olive','red'])
    #plt.title(Title)
    plt.show()
    #plt.savefig("Features_Engineering_Boxplot")
violin_boxplott(df=pandas_df_dummy)

# COMMAND ----------

## Calcul de la corrélation entre les variables (features)
corr=pandas_df_dummy.corr(method='pearson')
corr=corr.sort_values(by=["Death"],ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(15,20))
sns.barplot(x=corr.values, y=corr.index.values);
plt.title("Correlation Plot at State Level")
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### III.	PREPARATION DES DONNEES 

# COMMAND ----------

##variables  categorical, attribuer 1 pour male et 0 pour female 
mask_sex = {'male': 1, 'female': 0, 'NaN':0}


# COMMAND ----------

pandas_df_dummy = pandas_df_dummy.replace(mask_sex)
df = df.replace(mask_sex)

# COMMAND ----------

pandas_df_dummy.head()

# COMMAND ----------

#pandas_df_dummy = pandas_df_dummy.dropna()


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

# COMMAND ----------

#clean_dataset(pandas_df_dummy)

pandas_df_dummy = pandas_df_dummy.fillna(method='ffill')

pandas_df_dummy[pandas_df_dummy==np.inf]=np.nan
pandas_df_dummy.fillna(pandas_df_dummy.mean(), inplace=True)


#clean_dataset(pandas_df_dummy)

df = df.fillna(method='ffill')
df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Répartition de données en features X et Target Y

# COMMAND ----------

x = pandas_df_dummy.drop(['id', 'Death'], axis=1)

y = pandas_df_dummy['Death']

#x = x.values.astype(np.float)
#y = y.values.astype(np.float)

# COMMAND ----------

x.replace([np.inf, -np.inf], np.nan, inplace=True)
x.fillna(999999, inplace=True)

# COMMAND ----------

#x = x.dropna()

#np.isnan(x)  

#np.where(x.values >= np.finfo(np.float32).max)

# COMMAND ----------

# MAGIC %md 
# MAGIC Répartir les données en features x et y: la variable target 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Selection des variables

# COMMAND ----------

## Développer le modèle décision tree 
dt = DecisionTreeClassifier(max_depth=3)

# COMMAND ----------

dt.fit(x, y)

# COMMAND ----------

## Faire ressortir les 
dt_feat = pd.DataFrame(dt.feature_importances_, index=x.columns, columns=['feat_importance'])
dt_feat.sort_values('feat_importance').tail(8).plot.barh()
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC Feature importance est très important dans le domaine de  la modélisation machine learning. Il permet de comprendre quelles sont les variables qui contribuent le plus dans le modèle et aussi de pouvoir interpreter les résultats. 
# MAGIC 
# MAGIC Cette partie est cruciale car elle va permettre aux data scientist de pouvoir expliquer aisément les résultats. 
# MAGIC Les variables apparaissant dans les features importance : 
# MAGIC 
# MAGIC 
# MAGIC - Age est la variable la plus importante dans le modèle.
# MAGIC 
# MAGIC - time_til_admission : seconde variable 
# MAGIC 
# MAGIC - days_to_event : 
# MAGIC 
# MAGIC - ARDS 

# COMMAND ----------



X = pandas_df_dummy.drop(['id', 'Death', 'age', 'time_til_admission', 'days_to_event'], axis=1)

Y = pandas_df_dummy['Death']

# COMMAND ----------

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(999999, inplace=True)

# COMMAND ----------

## Développer le modèle décision tree 
dt1 = DecisionTreeClassifier(max_depth=3)



# COMMAND ----------

dt1.fit(X, Y)

# COMMAND ----------

## Faire ressortir les 
dt_feat1 = pd.DataFrame(dt1.feature_importances_, index=X.columns, columns=['feat_importance'])
dt_feat1.sort_values('feat_importance').tail(8).plot.barh()
plt.show()



# COMMAND ----------

# MAGIC %md 
# MAGIC Feature importance est très important dans le domaine de  la modélisation machine learning. Il permet de comprendre quelles sont les variables qui contribuent le plus dans le modèle et aussi de pouvoir interpreter les résultats. 
# MAGIC 
# MAGIC Cette partie est cruciale car elle va permettre aux data scientist de pouvoir expliquer aisément les résultats. 
# MAGIC Les variables apparaissant dans les features importance : 
# MAGIC  
# MAGIC 
# MAGIC - ARDS 
# MAGIC 
# MAGIC - Admission 
# MAGIC 
# MAGIC - other_unspecified 
# MAGIC 
# MAGIC - cough

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###  IV.	MODELISATION 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Description du process de modélisation
# MAGIC 
# MAGIC 
# MAGIC Nou testerons dans cette section plusieurs modèles de classification tels les SVM, les methodes ensembles (RF, adaboost,...), Reseaux de neurones etc... dans l'objectif de choisir le meilleur modèle. Puis nous optimiserons les hyperparametres des modèles qui nous semblent les plus performants. Enfin nous generons les courbes d'apprentissage afn d'évaluer le niveau d'apprentissage de ces modèles pour verifer l'overfitting où  l'underfitting. 

# COMMAND ----------

classifiers = {'Logistic Regression' : LogisticRegression(),
               'KNN': KNeighborsClassifier(),
               'Decision Tree': DecisionTreeClassifier(),
               'Random Forest': RandomForestClassifier(),
               'AdaBoost': AdaBoostClassifier(),
               'SVM': SVC()}

samplers = {'Random_under_sampler': RandomUnderSampler(),
            'Random_over_sampler': RandomOverSampler()}

drop_cols = ['id', 'dead', 'outcome2', 'outcome', 'time_til_recovery', 'time_til_death', 'age', 'time_til_admission', 'days_to_event']

# COMMAND ----------

def df_split(df, target='Death', drop_cols=drop_cols):
    df = df.drop(drop_cols, axis=1)
    df = df.fillna(999)
    x = df.drop(target, axis=1)
    y = df[target]    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)                          
    return x_train, x_test, y_train, y_test

def train_clfs(df, classifiers, samplers):
    
    x_train, x_test, y_train, y_test = df_split(df)
    
    names_samplers = []
    names_clfs = []
    results_train_cv_roc_auc = []
    results_train_cv_recall = []
    results_train_cv_accuracy = []
    results_test_roc_auc = []
    results_test_recall = []
    results_test_accuracy = []
    
    for name_sampler, sampler in samplers.items():
        print(f'Sampler: {name_sampler}\n')
        for name_clf, clf in classifiers.items():
            print(f'Classifier: {name_clf}\n')
            
            pipeline = Pipeline([('sampler', sampler),
                                 ('clf', clf)])
            
            cv_auc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='roc_auc') 
            cv_rec = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='recall')                                
            cv_acc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='accuracy')        

            pipeline.fit(x_train, y_train)        
            y_pred = pipeline.predict(x_test)
            
            names_samplers.append(name_sampler)
            names_clfs.append(name_clf)
            results_train_cv_roc_auc.append(cv_auc)
            results_train_cv_recall.append(cv_rec)
            results_train_cv_accuracy.append(cv_acc)
            results_test_roc_auc.append(roc_auc_score(y_test, y_pred))
            results_test_recall.append(recall_score(y_test, y_pred))
            results_test_accuracy.append(accuracy_score(y_test, y_pred))

            print(f'CV\t-\troc_auc:\t{round(cv_auc.mean(), 3)}')
            print(f'CV\t-\trecall:\t\t{round(cv_rec.mean(), 3)}')
            print(f'CV\t-\taccuracy:\t{round(cv_acc.mean(), 3)}')

            print(f'Test\t-\troc_auc:\t{round(roc_auc_score(y_test, y_pred), 3)}')         
            print(f'Test\t-\trecall:\t\t{round(recall_score(y_test, y_pred), 3)}')          
            print(f'Test\t-\taccuracy:\t{round(accuracy_score(y_test, y_pred), 3)}')      
            print('\n<-------------------------->\n')

    df_results_test = pd.DataFrame(index=[names_clfs, names_samplers], columns=['ROC_AUC', 'RECALL', 'ACCURACY'])
    #df_results_test = pd.DataFrame(index=[names_clfs, names_samplers], columns=['RECALL', 'ACCURACY'])
    df_results_test['ROC_AUC'] = results_test_roc_auc
    df_results_test['RECALL'] = results_test_recall
    df_results_test['ACCURACY'] = results_test_accuracy

    return df_results_test

# COMMAND ----------

df_results_test = train_clfs(df, classifiers, samplers)

# COMMAND ----------

df.dtypes

# COMMAND ----------

#pandas_df_dummy = pandas_df_dummy.drop(['id','_c0', 'dead', 'outcome2', 'outcome', 'time_til_recovery', 'time_til_death'], axis=1)

df['sex'] = pd.to_numeric(df['sex'])
df['age'] = pd.to_numeric(df['age'])
df['hospitalized'] = pd.to_numeric(df['hospitalized'])
df['time_til_admission'] = pd.to_numeric(df['time_til_admission'])


df['time_til_recovery'] = pd.to_numeric(df['time_til_recovery'])
df['time_til_death'] = pd.to_numeric(df['time_til_death'])
#df



# COMMAND ----------

# MAGIC %md 
# MAGIC #### Hyperparameters tuning

# COMMAND ----------

def train_xgb(df, clf):
    
    x_train, x_test, y_train, y_test = df_split(df)

    scale_pos_weight = len(df[df['Death'] == 0]) / len(df[df['Death'] == 1])

    param_grid = {'xgb__max_depth': [3, 4, 5, 6, 7, 8],
                  'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
                  'xgb__colsample_bytree': [0.6, 0.7, 0.8],
                  'xgb__min_child_weight': [0.4, 0.5, 0.6],
                  'xgb__gamma': [0, 0.01, 0.1],
                  'xgb__reg_lambda': [6, 7, 8, 9, 10],
                  'xgb__n_estimators': [150, 200, 300],
                  'xgb__scale_pos_weight': [scale_pos_weight]}

    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=100,
                                n_jobs=-1, verbose=2, cv=5,                            
                                scoring='roc_auc', random_state=42)

    rs_clf.fit(x_train, y_train)
    
    print(f'XGBOOST BEST PARAMS: {rs_clf.best_params_}')
    
    y_pred = rs_clf.predict(x_test)

    #df_results_xgb = pd.DataFrame(index=[['XGBoost'], ['No_sampler']], columns=['RECALL', 'ACCURACY'])
    df_results_xgb = pd.DataFrame(index=[['XGBoost'], ['No_sampler']], columns=['ROC_AUC', 'RECALL', 'ACCURACY'])

    df_results_xgb['ROC_AUC'] = roc_auc_score(y_test, y_pred)
    df_results_xgb['RECALL'] = recall_score(y_test, y_pred)
    df_results_xgb['ACCURACY'] = accuracy_score(y_test, y_pred)
    
    return df_results_xgb

# COMMAND ----------

df_results_xgb = train_xgb(df, xgb.XGBClassifier())

# COMMAND ----------

df_results = pd.concat([df_results_test, df_results_xgb])

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Evaluation des modèles 

# COMMAND ----------


df_plot = pd.concat([df_results.sort_values('ROC_AUC', ascending=False).head(3), 
                     df_results.sort_values('RECALL', ascending=False).head(3),
                     df_results.sort_values('ACCURACY', ascending=False).head(3)])

# COMMAND ----------

def plot_test(df, xlim_min, xlim_max):

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,12))
    color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'navy', 'turquoise', 'darkorange']

    df['ROC_AUC'].plot(kind='barh', ax=ax1, xlim=(xlim_min, xlim_max), title='ROC_AUC', color=color)
    df['RECALL'].plot(kind='barh', ax=ax2, xlim=(xlim_min, xlim_max), title='RECALL', color=color)
    df['ACCURACY'].plot(kind='barh', ax=ax3, xlim=(xlim_min, xlim_max), title='ACCURACY', color=color)
    plt.show()

# COMMAND ----------

plot_test(df_plot, 0.4, 1)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC *On peut constater que : 
# MAGIC 
# MAGIC - Pour le métrique : ROC-AUC 
# MAGIC 
# MAGIC 
# MAGIC      Les modèles (AdaBoost, Random_over_sampler),  (AdaBoost, Random_under_sampler),  (Logistic Regression, Random_over_sampler) présentent de beaux scores d'accuracy pour la courbe d'apprentissage comme pour la courbe de validation. 
# MAGIC     Mais nous risquerons d'être  en presence d'un cas d'over-fitting. 
# MAGIC     
# MAGIC - Pour Recall (Rappel)
# MAGIC     
# MAGIC     (AdaBoost, Random_over_sampler),  (AdaBoost, Random_under_sampler),  (Logistic Regression, Random_over_sampler) ont la valeur maximale de rappel et tend vers 1. 
# MAGIC 
# MAGIC 
# MAGIC - Pour l'accuracy 
# MAGIC     
# MAGIC   (SVM, Random_under_sampler),  (KNN, Random_over_sampler),  (XGBoost, No_sampler) ont les meilleurs scores  
# MAGIC 
# MAGIC 
# MAGIC Nos meilleurs modèles sont les suivants:
# MAGIC 
# MAGIC     - AdaBoost, Random_over_sampler
# MAGIC     
# MAGIC     - AdaBoost, Random_under_sampler 
# MAGIC     
# MAGIC     
# MAGIC     - Logistic Regression, Random_over_sampler
# MAGIC    

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Analyse de matrices de confusions à différents seuils de confiance 

# COMMAND ----------

def plot_confusion_matrix(y_test, y_pred, title='Confusion matrix'):
    
    cm = confusion_matrix(y_test, y_pred)
    classes = ['Patient non décédé', 'Patient décédé']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, )
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def train_clf_threshold(pandas_df_dummy, clf, sampler=None):
    thresholds = np.arange(0.1, 1, 0.1)
    
    x_train, x_test, y_train, y_test = df_split(pandas_df_dummy)
    
    if sampler:
        clf_train = Pipeline([('sampler', sampler),
                              ('clf', clf)])
        
    else:        
        clf_train = clf
            
    clf_train.fit(x_train, y_train)
    y_proba = clf_train.predict_proba(x_test)
    
    plt.figure(figsize=(20,20))

    j = 1
    for i in thresholds:
        y_pred = y_proba[:,1] > i

        plt.subplot(4, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test,y_pred)
        np.set_printoptions(precision=2)

        print(f"Threshold: {round(i, 1)} | Test Accuracy: {round(accuracy_score(y_test, y_pred), 2)}| Test Recall: {round(recall_score(y_test, y_pred), 2)} | Test Roc Auc: {round(roc_auc_score(y_test, y_pred), 2)}")

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, title=f'Threshold >= {round(i, 1)}')

# COMMAND ----------

train_clf_threshold(df, RandomForestClassifier(), sampler=RandomUnderSampler())

# COMMAND ----------

## Seuil de confiance : 90%


TN = 753
FP = 16
FN = 11
TP = 3

##  la sensibilité est le ratio du nombre de vrai positifs par le nombre total d'éléments positifs (y compris ceux déclarés faux par erreur).
## La spécificité c'est le ratio du nombre de vrai négatifs par le nombre total d'élément négatifs (y compris ceux déclarés vrai par erreur).

##

sensitivity = TP / float(FN + TP)

print("Sensibilité : %.2f" %  sensitivity)

specificity = TN / (TN + FP)

print("Spécificité  : %.2f" %  specificity)

precision = TP / float(TP + FP)

print("Précision  : %.2f" % precision)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Analyse de résultats 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC - Une Confusion Matrix (matrice de confusion) ou tableau de contingence est un outil permettant de mesurer les performances d’un modèle de Machine Learning en vérifiant notamment à quelle fréquence ses prédictions sont exactes par rapport à la réalité dans des problèmes de classification. 
# MAGIC 
# MAGIC  - A partir de la matrice de confusion, de métriques d'évaluation ont été extraites pour permettre l'analyse du modèle. 
# MAGIC  
# MAGIC  - Le rappel ("recall"  en anglais), ou sensibilité ("sensitivity" en anglais), est le taux de vrais positifs, c’est à dire la proportion de positifs que l’on a correctement identifiés. C’est la capacité de notre modèle à détecter tous les patients décédés. 21% 
# MAGIC  
# MAGIC  - la précision, c’est-à-dire la proportion de prédictions correctes parmi les points que l’on a prédits positifs. C’est la capacité de notre modèle à ne déclencher le traitement que pour un vrai malade potentiellement risqué c'est à dire pouvant décéder. 16% 
# MAGIC  
# MAGIC  - La spécificité ("specificity" en anglais), qui est le taux de vrais négatifs, autrement dit la capacité à détecter toutes les  situations où il n’y a pas de décès. C’est une mesure complémentaire de la sensibilité. 98% 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### V.	CONCLUSION

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ####  Les résultats sont assez mitigé après ce premier entrainement. On peut noter les résultats :
# MAGIC - la classification  RandomForestClassifier de résultats encourangeants en terme de sensibilité, précision, et specificité. 
# MAGIC 
# MAGIC 
# MAGIC - le modèle AdaBoost pourrait être une alternative. 
# MAGIC 
# MAGIC 
# MAGIC #### Prochaines étapes pour améliorer le modèle:
# MAGIC 
# MAGIC 
# MAGIC - Ajouter de nouvelles variables sur la base de nouvelles ingestions de bases de données plus historiques sur les patients. 
# MAGIC 
# MAGIC - Création, transformation et génération des nouvelles variables plus discriminantes.
# MAGIC 
# MAGIC - Tunner les hyperparamètres pour diminuer l'overfitting. 
# MAGIC 
# MAGIC - Ajouter d'autres critères d'évaluation des performances des algorithmes. 
# MAGIC 
# MAGIC - Itérer plusieurs modèles en fonction des produits
