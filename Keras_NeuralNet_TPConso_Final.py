# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   jupytext_formats: ipynb,py:light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.5.4
# ---

# # TP2 Prévision de consommation avec réseau de neurones
#
#
# <img src="pictures/Présentation_FormationIA_TPDeepLearning.png" width=1000 height=60>
#
# **Dans l'épisode précédent**  
#
# Nos modèles de régression par décomposition saisonnière du TP1 avec la librairie Prophet nous ont donné des premiers résultats et des premières intuitions sur notre problème de prévision de consommation pour le lendemain. 
#
# Nous avons pu analyser des profils de courbe de consommation au jour, à la semaine, au mois. Nous avons également observé la dépendance entre la consommation et la consommation retardée. Nous avons aussi vu l'impact des jours fériés. En intégrant ces différents facteurs nous sommes arrivés à une erreur moyenne de test de 4,1%.
#
# Nous avons ensuite utilisé de premiers modèles en machine learning pour apprendre par observations l'influence de différents contextes sur la consommation sans les décrires explicitement selon des lois. 
# Des difficultés se sont posées pour intégrer les variables météorologiques très dépendantes entre elles et pour intégrer un vecteur de consommation retardée.
#
# Avec l'approche classique exposée dans ce TP1, nous avons en particulier constaté le besoin d'une expertise et d'un travail autour des variables explicatives pour obtenir un modèle performant.
#
#

# **Aujourd'hui** 
#
# Nous allons de nouveau nous attaquer à ce sujet de la prévision de consommation nationale pour le lendemain, mais cette fois en utilisant un modèle de prévision par réseau de neurones. Nous allons exploiter leur capacité à capter ces phénomènes non-linéaires et interdépendants. Nous allons mettre en évidence le moindre besoin en feature engineering en travaillant directement à la granularité de la donnée,sans créer de variables agrégées ou transformées par de l'expertise.
#
# **Ce que vous allez voir dans ce second TP**
#
# - Un rappel de notre problème et récapitulatif des performances de nos modèles précédents
# - Une nouvelle méthode numérique pour préparer ses données et faciliter l'apprentissage: la normalisation
# - La création d'un premier réseau de neurones pour prédire la consommation dans 24h
# - L'utilisation de tensorboard pour observer en temps réel la courbe d'apprentissage du réseau de neurones
# - La création de modèles de plus en plus performants en intégrant davantage d'informations dans notre modélisation
# - L'évaluation des modèles sur 2 types de jeux de test
#
# __NB__ : Pour ce TP nous utiliserons Keras, une bibliothèque python de haut niveau qui appelle des fonctions de la librairie TensorFlow. D'autres librairies existent, Keras a été retenue en raison de sa facilité d'utilisation.

# # Dimensionnement en temps
# La durée estimée de ce TP est d'environ 1h30 :
# - 10-20 minutes pour charger et préparer les données pour les réseaux de neurones 
# - 10-20 minutes pour entrainer un premier modèle de réseau de neurones, en examiner le code implémentant ce réseau de neurones
# - Le reste pour jouer et tenter d'améliorer la qualité de la prédiction avec de nouvelles variables explicatives, ou en choisissant d'autres hyper-paramètres. 

# # I) Préparation des données

# #%autosave 0
import sys
print(sys.path)

# ## Chargement des librairies nécessaires

# +
import os
import numpy as np
import pandas as pd
import random
import datetime
from IPython.display import SVG

import plotly
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# sklearn est la librairie de machine learning en python et scipy une librairie statistiques
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy import stats

#######
# Keras est la librairie que nous utilisons pour se créer des modèles de réseau de neurones
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, Dense
from keras.callbacks import TensorBoard
from time import time

import zipfile


K.set_image_data_format('channels_last')

# %matplotlib inline
# -

# ## Localisation des données

# +
## Choix du répertoire de travail "data_folder" dans lequel toutes les fichiers csv seront entreposés

# a actualiser avec le repertoire par defaut sur le serveur
data_folder = os.path.join(os.getcwd(),"data")

### Petite vérification
print("Mon repertoire est : {}".format(data_folder))
print("Fichiers contenus dans ce répertoire :")
for file in os.listdir(data_folder):
    print(" - " + file)
# -

# ## I) Récupération et préparation des données
#
# Dans cette partie nous allons charger les fichiers csv nécessaires pour l'analyse, puis les convertir en data-frame python. Les données de base à récupérer sont :
# - la base de données issues du TP1 (Les historiques de consommation, leur lag, les données météo en température, leur lag, les jours feriés) 
#
# En terme de transformation des données pour mieux les préparer:
#
# - nous allons aussi voir comment normaliser les données, une transformation souvent bien utile en pratique pour une meilleure convergence numérique. 
#
# Cela vient compléter les transformations vu précédemment pour les données calendaires, et aussi la transformation "one-hot" pour les données catégorielles

# ### Récupération de nos variables explicatives XInput

# #### Les données d'entrée sont encryptées.
#
# En effet les données météo sont confidentielles. Pour les lire vous avez besoin d'un mot de passe qui ne peut vous être donné que dans le cadre d'un travail au sein de RTE
#

password=None#

# +
Xinput_zip = os.path.join(data_folder, "Xinput.zip")
zfXinput = zipfile.ZipFile(Xinput_zip)#.extractall(pwd=bytes(password,'utf-8'))
zfXinput.setpassword(bytes(password,'utf-8'))
Xinput = pd.read_csv(zfXinput.open('Xinput.csv'),sep=",",engine='c',header=0)

#Xinput_csv = os.path.join(data_folder, "Xinput.csv")
#Xinput = pd.read_csv(Xinput_csv, sep=",", engine='c', header=0) 
Xinput['ds']=pd.to_datetime(Xinput['ds'])

Xinput = Xinput.drop(['Unnamed: 0'], axis=1)
print(Xinput.shape)

print("voici toutes les variables explicatives que l'on peut utiliser")
Xinput.columns.get_values()
#print(*Xinput.columns, sep='\n')
#print(Xinput.columns,end=None)
# -

# **Pour Rappel**
#
# Les transformations calendaires avaient été appliquées sur la variable de date 'ds' pour extraire les mois et les jours.
#
# Les transformations "one hot" avaient été utilisées pour transformer les variables catégorielles 'month'(1 à 12) et 'hour' (1 à 24) en 'month_1...month_12' et 'hour_0...hour_23'
#
# Les variables retardées d'une journée ont été intégrées

#####
#suppression de lignes en doublons si besoin
u, indices = np.unique(Xinput['ds'].get_values(),return_index=True) #on recupere des timeStamps en double ce qui peut arriver après jointure des données
Xinput=Xinput.iloc[indices]
Xinput.reset_index(drop=True)
Xinput.head(5)

# ### Récupération de nos variables à prédire: la consommation française

# +
Yconso_csv = os.path.join(data_folder, "Yconso.csv")
Y = pd.read_csv(Yconso_csv, sep=",", engine='c', header=0) 
Y['ds']=pd.to_datetime(Y['ds'])

#####
#suppression de lignes en doublons si besoin
u, indices = np.unique(Y['ds'].get_values(),return_index=True)
Y=Y.iloc[indices]
Y.reset_index(drop=True)

Y = Y.drop(['Unnamed: 0'], axis=1)
Y.head(10)
# -

# ### Tronquer les données

# Il faut tronquer les données car nous avons rajouté comme variables explicatives des lags sur la consommation et la météo qui n'ont pas de valeurs au début de l'historique
#

# +
nbJourlagRegresseur=7 #du fait du lag de 1 semaine pour la consommation que nous utilisons
dateStart = Xinput.iloc[0]['ds']
DateStartWithLag = dateStart + pd.Timedelta(str(nbJourlagRegresseur) + ' days')

Xinput = Xinput[(Xinput.ds >= DateStartWithLag)]
Xinput=Xinput.reset_index(drop=True)

Y = Y [(Y.ds >= DateStartWithLag)]
Y=Y.reset_index(drop=True)
Y.head(10)
# -

# ## Normalisation des données
# En théorie, la normalisation des données d'entrée n'est pas indispensable pour entrainer un réseau de neurones. En effet, on devrait apprendre des poids et biais plus ou moins importants pour équilibrer les contributions des différentes variables explicatives en entrée. 
#
# Cependant en pratique, normaliser les données d'entrée permet généralement d'obtenir un apprentissage plus rapide du réseau de neurones.
#
# **Question**: comment l'expliquez-vous ?

# +
# Normalisation de la meteo et du lag de la conso

#on instancie un modèle qui va normaliser les données, en faisant l'hypothèse que la distribution suit une loi normale pour chacune d'entre elles: 
#ce "scaler a donc pour paramètres la moyenne de la distribution et son écart type
scaler = StandardScaler(with_mean=True,with_std=True)

#on enumère les variables à standardiser indépendamment
colsToScale=[s for s in Xinput.columns.get_values() if (('Th' in s) )]
colsToScale.append('lag1D')

#on se crée un nouveau X qui contiendra les variables standardisées:Xinput_scaled
Xinput_scaled=Xinput.copy()

#on apprend nos paramètres: moyenne et écart type
scalerfit=scaler.fit(Xinput[colsToScale])

#on normalise nos données au vu de ces paramètres
new_ = scalerfit.transform(Xinput_scaled[colsToScale])
for i, ncol_name in enumerate(colsToScale):
    Xinput_scaled[ncol_name]= new_[:,i]
    
# -

# Attention: pour etre rigoureux, il faudrait calculer moyenne et ecart type uniquement sur les données d'entrainement pour renormaliser l'ensemble des données et apprendre ces paramètres de normalisation. Les donnees de tests elles devant restées complètement masquées pendant l'apprentissage.
# Ici, on normalise sur l'ensemble du jeu par soucis de simplification. Nous séparerons notre jeu en jeu de test et d'entrainement par la suite
#
# Visualisons une distribution après normalisation

# +
nomVariable=colsToScale[10]
print("voici les distrinutions initiale et normalisée de la station:"+nomVariable)

plt.subplot(1, 2, 1)
plt.hist(Xinput[colsToScale[10]], bins=200)
plt.xlabel('T(°C)')
plt.title("distribution initiale de:"+nomVariable)

plt.subplot(1, 2, 2)
plt.hist(Xinput_scaled[colsToScale[10]], bins=200,color='green')
plt.xlabel('T(°C)')
plt.title("distribution normalisée de:"+nomVariable)
plt.tight_layout(w_pad=5)
plt.show()
# -

# ## II) Création de jeux d'apprentissage et de test
#
# **Question** : A quoi servent les jeux d'entrainement, de validation, et de test ?
#
# On se crée et on extrait 2 jeux de test de type différent, les données restantes constituant notre jeu d'entrainement-validation.
# Ces 2 jeux de tests sont établis ainsi:
#  - sélectionner une date de coupure (mai 2017) pour évaluer notre modèle, appris sur un historique, sur une période future.
#  - sélection de quelques dates au hasard en été et en hiver avec la fonction _selectDatesRandomly_ pour établir la performance de notre modèle sur des journées thermosensibles.
#

def selectDatesRandomly(XwithDs, quarter, nDays, seed=10):
    joursQuarter = np.unique(XwithDs.loc[np.where(XwithDs.ds.dt.quarter == quarter)].ds.dt.date)
    np.random.seed(seed)
    joursSelected = np.random.choice(joursQuarter, replace=False, size=nDays)
    return joursSelected

# +
# 1er jeu de test pour une séquence non vue, a partir du 1er mai 2017 par exemple
# 2eme jeu de test pour des jours dans l'historique du jeu d'entrainement, en particulier l'hiver et l'été pour capter l'effet de la température

def prepareDataSetEntrainementTest_TP2(Xinput, Yconso, dateDebut, dateRupture, nbJourlagRegresseur=0, njoursEte=10, njoursHiver=10):
    
    dateStart = Xinput.iloc[0]['ds']
    
    #les problèmes de données dues au lag sont maintenant gérés lorsque l'on charge les données
    #DateStartWithLag = dateStart + pd.Timedelta(str(nbJourlagRegresseur) + ' days')  # si un a un regresseur avec du lag, il faut prendre en compte ce lag et commencer l'entrainement a la date de debut des donnees+ce lag
   
    #On crée notre jeu de test "future" selon la date de rupture souhaitée
    XinputTest1 = Xinput[(Xinput.ds >= dateRupture)]
    XinputTrain = Xinput[(Xinput.ds < dateRupture) & (Xinput.ds >= dateDebut)] #& (Xinput.ds > DateStartWithLag) 
    YconsoTrain = Yconso[(Yconso.ds < dateRupture) & (Xinput.ds >= dateDebut)] #& (Yconso.ds > DateStartWithLag)
    YconsoTest1 = Yconso[(Yconso.ds >= dateRupture)]
    
    XinputTrain = XinputTrain.reset_index(drop=True)
    YconsoTrain = YconsoTrain.reset_index(drop=True)
    
    #On crée notre jeu de test "thermosensible"
    
    #On sélectionne des jours d'hiver et d'été
    joursHiverSelectionne = selectDatesRandomly(XinputTrain, 4, njoursHiver, seed=10)
    print("Les jours d'hiver du jeu de test sont : ")
    print(joursHiverSelectionne)
    
    joursEteSelectionne=selectDatesRandomly(XinputTrain, 2, njoursEte, seed=20)
    print("Les jours d'été du jeu de test sont : ")
    print(joursEteSelectionne)
    
    joursSelectionne = np.append(joursHiverSelectionne, joursEteSelectionne)
    
    XinputTest2 = XinputTrain[np.in1d(XinputTrain.ds.dt.date, joursSelectionne)]
    YconsoTest2 = YconsoTrain[np.in1d(YconsoTrain.ds.dt.date, joursSelectionne)]
    
    XinputTrain = XinputTrain[~np.in1d(XinputTrain.ds.dt.date, joursSelectionne)]
    YconsoTrain = YconsoTrain[~np.in1d(YconsoTrain.ds.dt.date, joursSelectionne)]
    
    XinputTrain = XinputTrain.reset_index(drop=True)
    YconsoTrain = YconsoTrain.reset_index(drop=True)
    XinputTest1 = XinputTest1.reset_index(drop=True)
    YconsoTest1 = YconsoTest1.reset_index(drop=True)
    XinputTest2 = XinputTest2.reset_index(drop=True)
    YconsoTest2 = YconsoTest2.reset_index(drop=True)
    
    XinputTrain = XinputTrain.drop(['ds'],axis=1)
    XinputTest1 = XinputTest1.drop(['ds'],axis=1)
    XinputTest2 = XinputTest2.drop(['ds'],axis=1)
    #indicesJoursEte=np.where(YconsoTrain.ds.dt.quarter==2)
    #joursHiver=
    #XinputTest2=Xinput[(Xinput.ds>=dateRupture)]
    
    #XinputTrain.reset_index(drop=True)
    #YconsoTrain.reset_index(drop=True)
    
    return XinputTrain, XinputTest1, XinputTest2, YconsoTrain, YconsoTest1, YconsoTest2

# +
dateRupture = datetime.datetime(year=2017, month=5, day=1)  # début du challenge prevision de conso
dateDebut = datetime.datetime(year=2013, month=1, day=1)  # début du début de l'historique
nbJourlagRegresseur = 1

XinputTrain, XinputTest1, XinputTest2, YconsoTrain, YconsoTest1, YconsoTest2 = prepareDataSetEntrainementTest_TP2(Xinput, Y,dateDebut, dateRupture, nbJourlagRegresseur)
# -

print('la taille de l échantillon XinputTrain est:' + str(XinputTrain.shape[0]))
print('la taille de l échantillon XinputTest1 est:' + str(XinputTest1.shape[0]))
print('la taille de l échantillon XinputTest2 est:' + str(XinputTest2.shape[0]))
print('la taille de l échantillon YconsoTrain est:' + str(YconsoTrain.shape[0]))
print('la taille de l échantillon YconsoTest1 est:' + str(YconsoTest1.shape[0]))
print('la taille de l échantillon YconsoTest2 est:' + str(YconsoTest2.shape[0]))
print('la proportion de data d entrainement est de:' + str(YconsoTrain.shape[0] / (YconsoTrain.shape[0] +YconsoTest2.shape[0]+ YconsoTest1.shape[0])))

XinputTrain.head()

print('Nombre de variables explicatives potentielles: ' + str(XinputTrain.shape[1]))
print('Nombre de variables explicatives potentielles: ' + str(XinputTest1.shape[1]))
print('Nombre de variables explicatives potentielles: ' + str(XinputTest2.shape[1]))

# # III) Création d'un modèle de réseau de neurones
#
# Jusqu'ici, nous avons importé nos données. Nous les avons ensuite préparées pour les fournir au réseau de neurones (one-hot encoding, normalisation). Nous avons également créé nos jeux d'entrainement et de test.
#
# Il est maintenant l'heure de se construire un réseau de neurones, de l'entrainer, et de lui faire faire des prédictions !
#
# **Cette partie est générique et indépendante de notre problématique de prévision de consommation**
#
# <img src="pictures/FirstNeuralNetwork.jpeg" width=700 height=60>

# ## Deux fonctions bien utiles
#
# Nous allons commencer par implémenter deux fonctions que nous appellerons pour chacun des modèles que nous allons tester:
# - Fonction 1: newKerasModel, pour instancier un modèle de réseau de neurone avant apprentissage
# - Fonction 2: plotYourNeuralNet, pour visualiser un réseau de neurones

# ### Création d'une architecture de réseau de neurones

def newKerasModel(nInputs,nOutput=1, hLayers = None):
    """      
    arguments
        - nInputs : le nombre de features en entrée
        - nOutput : le nombre de sorties, variables à prédire
        - hLayers : un vecteur de couches cachées précisant la taille de chaque couchees
        
    returns
        - un objet de type Model 
    """
    model = Sequential()
    if(hLayers==None):
        hLayers = [nInputs, nInputs, nInputs, nInputs, nInputs, nInputs]
        
    nHiddenLayers=len(hLayers)
    
    model.add(Dense(hLayers[0], input_dim=nInputs, activation='relu'))
    for l in range(nHiddenLayers-1):
        model.add(Dense(hLayers[l+1], activation='relu'))

    # Pour une régression, la fonction d'activation finale est simplement la fonction identité
    model.add(Dense(nOutput, activation='linear'))  
    
    return model

# ### Inspection de l'architecture d'un reseau de neurones
# On se créé un réseau avec un certains nombre de couches qui peuvent chacune avoir différentes dimensions. On peut ensuite inspecter les dimensions et le nombre de paramètres de ce réseau avec la méthode summary de Keras. 

#on se crée un réseau de neurones avec un certains nombre d'entrées et sorties
nInputs = 8 #un choix raisonnable pour visualiser ce modèle ensuite
nOutput=1
#hiddenLayers=[nInputs,round(nInputs/2),round(nInputs/2)]
hiddenLayers=[nInputs,nInputs,nInputs,nInputs,nInputs]
modelVide = newKerasModel(nInputs,nOutput,hLayers=hiddenLayers)
modelVide.summary()

# Créons-nous maintenant une fonction pour dessiner ce réseau de neurones

# +
# %run auxiliairyMethodsTP2.py #on charge un fichier où sont définies nos fonctions auxilaires non détaillées dans ce TP
import pydot
from IPython.display import Image

def plotYourNeuralNet(model):
    layersModel=[model.input_shape[1]]
    for layer in model.layers:
        layersModel.append(layer.get_output_at(0).get_shape().as_list()[1])

    plotNeuralNet(layersModel)
    (graph,) =pydot.graph_from_dot_file('out.dot')
    
    fileNameImage='yourNeuralNet.png'
    graph.write_png(fileNameImage, prog='dot')
    img = Image(fileNameImage)
    display(img)


# -

# Visualisons le reseau de neurone test créé précédemment

plotYourNeuralNet(modelVide)

# ## IV) Un premier modèle de réseau de neurones: variables calendaires + lag conso

# ## Choix des variables explicatives
#
# pour ce TP, nous avons un jeu d'entrée X contenant beaucoup de variables. Afin de commencer par un modèle simple, nous allons élaguer ce X pour réduire le nombre de features en entrée. Dans ce TP, nous allons donc lister les colonnes à retirer des datasets X initialisés ci-dessus.
#
# Pour un cas d'étude réel, une approche pragmatique serait de commencer par se créer un premier X simple, de voir les performances du modèle, puis ensuite d'incorporer de plus en plus de features dans le X pour évaluer la progression des performances de nos modèles.
# Toutefois en deep learning de premiers essais sont souvent réalisés en utilisant en entrée toute l'information disponible du fait de leur capacité à "digérer" la donnée, en se nourrissant d'informations redondantes.
# D'un point de vue pédagogique, nous allons commencer avec la première approche.
#
# Pour le premier réseau de neurones que nous allons entrainer, nous allons simplement garder les variables calendaires de Xinput.

# Initialement
XinputTrain.head()

# On sélectionne les variables que l'on souhaite conserver en précisant simplement à quelle catégorie elles appartiennent.

# +
#on précise le type de variable que l'on veut conserver par une abbréviation
colsType= ['lag1D','month','hour','dayOfWeek','JoursFeries']
#any(ext in colsType for ext in extensionsToCheck)
colsToKeep=[s for s in Xinput.columns.get_values() if any(cs in s for cs in colsType)]

if('ds' in colsToKeep):#cette variable a été décomposée en mois, heure, jour et n'est plus une variable d'intérêt
    colsToKeep.remove('ds')

# -

# On restreint nos jeux d'entrainement et de tests à ces variables

X = XinputTrain[colsToKeep]
XTest1 = XinputTest1[colsToKeep]
XTest2 = XinputTest2[colsToKeep]

# Après élagage des variables
print(X.columns)
print(X.shape)
print(XTest1.shape)
print(XTest2.shape)

# ## Création du réseau de neurones et hyper-paramétrage
# Un réseau de neurones profond est constuitué d'un certains nombre de couches, chacune portant un certain nombre de neurones. Ce sont 2 hyperparamètres que vous pouvez faire varier et qui vous permettront d'obtenir un apprentissage plus ou moins précis, en utilisant plus ou moins de puissance de calcul.
#
# Le "learning rate" de l'optimiseur est également un hyperparamètre qui influencera la convergence et la vitesse de convergence de l'apprentissage, où l'on cherche à optimiser notre modèle pour minimiser l'erreur de prédiction. 

nInputs = X.shape[1]#nombre d'entrées du modèle
first_model = newKerasModel(nInputs)

first_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

# # Tensorboard
# c'est un utilitaire de tensorflow qui permet de visualiser en temps réel les courbes d'apprentissage des réseau de neurones et est donc utile pour arrêter l'apprentissage si les progrès sont faibles.
#
# En particulier, vous pouvez vous intéresser à la courbe de l'erreur (loss) d'entrainement et de validation pour visualiser la progression de l'apprentissage et une tendance au surapprentissage en fin d'apprentissage.
#
# **Pour ouvrir une fenêtre tensorboard, revenez sur la page d'accueil de Jupyter puis cliquez sur New (en haut à droite) et enfin sur Tensorboard**
# Une fenêtre pop-up doit s'ouvrir. Si elle est bloquée, autorisez son ouverture.

# Conserver ses imports redondants ici car sinon des problèmes ont été constatés
from keras.callbacks import TensorBoard
from time import time
tensorboard = TensorBoard(log_dir="logs/{}".format("first_model_" +str(time())))

# ## Entrainement
#
# La cellule suivante peut prendre un peu de temps à s'exécuter. On reconnait là la méthode fit commune à chaque modèle de machine learning pour entraîner son modèle.

# +
# Fit the model

#en terme de paramètres:
#epoch: on précise le nombre d'epochs (le nombre de fois que l'on voit le jeu d'apprentissage en entier)
#batch size: le nombre d'exemples sur lequel on fait un "pas" d'apprentissage parmi tout le jeu
#validation_split: la proportion d'exemples que l'on conserve pour notre jeu de validation
#callbacks: pour appeler des utilitaires/fonctions externes pour récupérer des résultats
first_model.fit(X, YconsoTrain['y'], epochs=100, batch_size=100, validation_split=0.1, callbacks=[tensorboard])
# -

# ## Evaluation de la qualité du modèle

predictionsTrain = first_model.predict(X).reshape(-1)
predictionsTest1 = first_model.predict(XTest1).reshape(-1)
predictionsTest2 = first_model.predict(XTest2).reshape(-1)
print(predictionsTrain)

# +
evaluation(YconsoTrain, YconsoTest1, predictionsTrain, predictionsTest1)
plt.plot(YconsoTest1['ds'], YconsoTest1['y'], 'b')
plt.plot(YconsoTest1['ds'], predictionsTest1, 'r')

plt.title("Evolution de l'erreur sur le test 1")
plt.show()

# +
evaluation(YconsoTrain, YconsoTest2, predictionsTrain, predictionsTest2)

plt.plot(YconsoTest2['y'], 'b')
plt.plot(predictionsTest2, 'r')

plt.title("Evolution de l'erreur sur le test2")
plt.show()
# -

# L'erreur est ici comparable à celle des autres modèles en machine Learning (random forest, xgboost). Cela peut nous conforter dans le fait que notre réseau de neurones s'est créé de bonnes représentations pour ces variables calendaires. 
#
# La différence en performance peut devenir plus flagrante lorsque l'on intègre des variables à une maille très granulaire (les pixels d'une images, la température dans toutes les villes de France) avec une forte interdépendance.

ErreursTest1, ErreurMoyenneTest, ErreurMaxTest, RMSETest = modelError(YconsoTest1, predictionsTest1)
num_bins=100
plt.hist(ErreursTest1, num_bins)
plt.show()

# Pour inspecter dynamiquement des visualisations, la librairie plotly se révèle très utile.
# Ci-dessous vous pouvez identifier les jours et heures qui présentent les erreurs les plus importantes pour ensuite imaginer ce qui a pu pêcher.

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, iplot_mpl
init_notebook_mode(connected=True)
iplot([{"x": YconsoTest1['ds'], "y": ErreursTest1}])

# ## V) Un deuxième modèle : variables calendaires + lag conso normalisé!

# ## Choix des variables explicatives
#
# pour ce TP, nous avons un jeu d'entrée X contenant beaucoup de variables. Afin de commencer par un modèle simple, nous allons élaguer ce X pour réduire le nombre de features en entrée. Dans ce TP, nous allons donc lister les colonnes à retirer des datasets X initialisés ci-dessus.
#
# Pour un cas d'étude réel, une approche plus logique serait de commencer par se créer un premier X simple, de voir les performances du modèle, puis ensuite d'incorporer de plus en plus de features dans le X.
#
# Pour le premier réseau de neurones que nous allons entrainer, nous allons simplement garder les variables calendaires de Xinput.

XinputTrain, XinputTest1, XinputTest2, YconsoTrain, YconsoTest1, YconsoTest2 = prepareDataSetEntrainementTest_TP2(Xinput_scaled, Y,dateDebut, dateRupture, nbJourlagRegresseur)

# Initialement
XinputTrain.head()

# On sélectionne les variables que l'on souhaite conserver en précisant simplement à quelle catégorie elles appartiennent.

# +
#on précise le type de variable que l'on veut conserver par une abbréviation
colsType= ['lag1D','month','hour','dayOfWeek','JoursFeries']
#any(ext in colsType for ext in extensionsToCheck)
colsToKeep=[s for s in Xinput.columns.get_values() if any(cs in s for cs in colsType)]

if('ds' in colsToKeep):#cette variable a été décomposée en mois, heure, jour et n'est plus une variable d'intérêt
    colsToKeep.remove('ds')
# -

X = XinputTrain[colsToKeep]
XTest1 = XinputTest1[colsToKeep]
XTest2 = XinputTest2[colsToKeep]

# Après élagage
print(X.columns)
print(X.shape)
print(XTest1.shape)
print(XTest2.shape)

# ## Création du réseau de neurones et hyper-paramétrage
# Un réseau de neurones profond est constuitué d'un certains nombre de couches, chacune portant un certain nombre de neurones. Ce sont 2 hyperparamètres que vous pouvez faire varier et qui vous permettront d'obtenir un apprentissage plus ou moins précis, en utilisant plus ou moins de puissance de calcul.
#
# Le "learning rate" de l'optimiseur est également un hyperparamètre qui influencera la convergence et la vitesse de convergence de l'apprentissage, où l'on cherche à optimiser notre modèle pour minimser l'erreur de prédiction. 

nInputs = X.shape[1]
first_model_scaled = newKerasModel(nInputs)

first_model_scaled.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

# # Tensorboard
# c'est un utilitaire de tensorflow qui permet de visualiser en temps réel les courbes d'apprentissage des réseau de neurones et est donc utile pour arrêter l'apprentissage si les progrès sont faibles.

# Ajouter la ligne ci-dessous lève le problème
from keras.callbacks import TensorBoard
from time import time
tensorboard = TensorBoard(log_dir="logs/{}".format("first_model_scaled" +str(time())))

# ## Entrainement
#
# La cellule suivante peut prendre un peu de temps à s'exécuter.

# Fit the model
first_model_scaled.fit(X, YconsoTrain['y'], epochs=100, batch_size=100, validation_split=0.1, callbacks=[tensorboard])

# ## Evaluation de la qualité du modèle

predictionsTrain = first_model_scaled.predict(X).reshape(-1)
predictionsTest1 = first_model_scaled.predict(XTest1).reshape(-1)
predictionsTest2 = first_model_scaled.predict(XTest2).reshape(-1)
print(predictionsTrain)

# +
evaluation(YconsoTrain, YconsoTest1, predictionsTrain, predictionsTest1)
plt.plot(YconsoTest1['ds'], YconsoTest1['y'], 'b')
plt.plot(YconsoTest1['ds'], predictionsTest1, 'r')

plt.title("Evolution de l'erreur sur le test 1")
plt.show()

# +
evaluation(YconsoTrain, YconsoTest2, predictionsTrain, predictionsTest2)

plt.plot(YconsoTest2['y'], 'b')
plt.plot(predictionsTest2, 'r')

plt.title("Evolution de l'erreur sur le test2")
plt.show()
# -

# L'erreur est ici comparable à celle de Prophet (5,8%) qui modèle nativement la notion de temps. Cela peut nous conforter dans le fait que notre réseau de neurones s'est créé de bonnes représentations pour ces variables calendaires

ErreursTest1, ErreurMoyenneTest, ErreurMaxTest, RMSETest = modelError(YconsoTest1, predictionsTest1)
num_bins=100
plt.hist(ErreursTest1, num_bins)
plt.show()


init_notebook_mode(connected=True)
iplot([{"x": YconsoTest1['ds'], "y": ErreursTest1}])


# ### Discussion:
# On observe qu'avec des données d'entrées normalisées l'apprentissage converge plus rapidement et même mieux !

# # A vous de jouer, faites fonctionner vos neurones :
#
# Il peut y avoir différents objectifs de performance selon le besoin. En général on cherche un compromis entre la précision du modèle et la puissance de calcul nécessaire pour entraîner et faire tourner ce modèle.
#
# Nous décernerons 2 récompenses!
#
# - Défi 1: le modèle le plus précis.
# - Défi 2: le modèle le plus frugal moyennant une perte de précision.
# Les critères sont encore à affiner ensemble!
#
# <img src="pictures/we-need-you.png" width=500 height=60>
#
#

# # entrainez et testez 2 nouveaux modèles avec de nouvelles variables choisies
#
# N'hésitez pas à vous inspirer par le code ci-dessus ;-)
# venez partager vos investigations sur cette google sheet : https://docs.google.com/spreadsheets/d/1oIx8jjzIh7Ugp3ZJMCOEwns6KCJxo4ua_jW5hIvjjFI/edit?usp=sharing

# # Votre modèle 

# ## Rappel des variables explicatives à disposition

# Initialement
XinputTrain.columns.get_values()

#ajout variable JoursFeries_J_1
Xinput['JoursFeries_J_1']=Xinput['JoursFeries'].shift(24)
Xinput['JoursFeries_J_1']=Xinput['JoursFeries_J_1'].fillna(0)
Xinput_scaled['JoursFeries_J_1']=Xinput['JoursFeries_J_1']
XinputTrain, XinputTest1, XinputTest2, YconsoTrain, YconsoTest1, YconsoTest2 = prepareDataSetEntrainementTest_TP2(Xinput_scaled, Y,dateDebut, dateRupture, nbJourlagRegresseur)

# ## Choix des variables explicatives

# On sélectionne les variables que l'on souhaite conserver en précisant simplement à quelle catégorie elles appartiennent.

# +
#########
#TO DO
#Préciser le type de variable que l'on veut conserver par une abbréviation
colsType= None

#Fin TO DO
#########

colsToKeep=[s for s in Xinput.columns.get_values() if any(cs in s for cs in colsType)]

if('ds' in colsToKeep):#cette variable a été décomposée en mois, heure, jour et n'est plus une variable d'intérêt
    colsToKeep.remove('ds')
# -

X = XinputTrain[colsToKeep]
XTest1 = XinputTest1[colsToKeep]
XTest2 = XinputTest2[colsToKeep]

# Après élagage
print(X.columns)
print(X.shape)
print(XTest1.shape)
print(XTest2.shape)

# ## Création du réseau de neurones, hyper-paramétrage

nInputs = X.shape[1]
votre_model_scaled_1 = newKerasModel(nInputs)

votre_model_scaled_1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

# # Tensorboard
# c'est un utilitaire de tensorflow qui permet de visualiser en temps réel les courbes d'apprentissage des réseau de neurones et est donc utile pour arrêter l'apprentissage si les progrès sont faibles.

# Ajouter la ligne ci-dessous lève le problème
from keras.callbacks import TensorBoard
from time import time
tensorboard = TensorBoard(log_dir="logs/{}".format("votre_model_scaled_1_" +str(time())))

# ## Entrainement
#
# La cellule suivante peut prendre un peu de temps à s'exécuter.

# Fit the model
votre_model_scaled_1.fit(X, YconsoTrain['y'], epochs=150, batch_size=100, validation_split=0.1, callbacks=[tensorboard])

# ## Evaluation de la qualité du modèle

predictionsTrain = votre_model_scaled_1.predict(X).reshape(-1)
predictionsTest1 = votre_model_scaled_1.predict(XTest1).reshape(-1)
predictionsTest2 = votre_model_scaled_1.predict(XTest2).reshape(-1)
print(predictionsTrain)

# +
evaluation(YconsoTrain, YconsoTest1, predictionsTrain, predictionsTest1)
plt.plot(YconsoTest1['ds'], YconsoTest1['y'], 'b')
plt.plot(YconsoTest1['ds'], predictionsTest1, 'r')

plt.title("Evolution de l'erreur sur le test 1")
plt.show()

# +
evaluation(YconsoTrain, YconsoTest2, predictionsTrain, predictionsTest2)

plt.plot(YconsoTest2['y'], 'b')
plt.plot(predictionsTest2, 'r')

plt.title("Evolution de l'erreur sur le test2")
plt.show()
# -

# L'erreur est ici comparable à celle de Prophet (5,8%) qui modèle nativement la notion de temps. Cela peut nous conforter dans le fait que notre réseau de neurones s'est créé de bonnes représentations pour ces variables calendaires

ErreursTest1, ErreurMoyenneTest, ErreurMaxTest, RMSETest = modelError(YconsoTest1, predictionsTest1)
num_bins=100
plt.hist(ErreursTest1, num_bins)
plt.show()

init_notebook_mode(connected=True)
iplot([{"x": YconsoTest1['ds'], "y": ErreursTest1}])


# # Pour aller encore plus loin
#
# Le modèle ci-dessus peut être rendu encore plus performant par exemple en considérant des features comme "jour d'avant vacances", "jour d'après vacances"... 
#
# Passer du temps à tuner les hyper-paramètres serait certainement bénéfique aussi.
#
# De manière assez surprenante, élargir le réseau de neurones pour prédire les consommations régionales peut également améliorer la qualité de la prédiction de l'échelle nationale. C'est l'idée du multi-tasking.
#
# On pourra également considérer en sortie du modèle non pas la prédiction pour juste 24 heures plus tard, mais plutôt pour une plage horaire  
# [1 heure plus tard, ..., 24 heures plus tard]. Ceci permet de capter des dynamiques.
#
#
# Des pistes:
# insérer le lag Jours_Feries_J-1
# voir si un intérêt à passer toutes les stations ou seulement la température France avec expertise sur importance des stations (en fait ça suffit avec les stations fournies!)
#




