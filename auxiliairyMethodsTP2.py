import os
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error

def plotNeuralNet(layers):
    with open('out.dot', 'w') as f:
        layers_str = ["Input"] + ["Hidden"] * (len(layers) - 2) + ["Output"]
        layers_col = ["none"] + ["none"] * (len(layers) - 2) + ["none"]
        layers_fill = ["black"] + ["gray"] * (len(layers) - 2) + ["black"]

        penwidth = 15
        font = "Hilda 10"

        print("digraph G {",file=f)
        print("\tfontname = \"{}\"".format(font),file=f)
        print("\trankdir=LR",file=f)
        print("\tsplines=line",file=f)
        print("\tnodesep=.08;",file=f)
        print("\tranksep=1;",file=f)
        print("\tedge [color=black, arrowsize=.5];",file=f)
        print("\tnode [fixedsize=true,label=\"\",style=filled," + \
              "color=none,fillcolor=gray,shape=circle]\n",file=f)

        # Clusters
        for i in range(0, len(layers)):
            print(("\tsubgraph cluster_{} {{".format(i)),file=f)
            print(("\t\tcolor={};".format(layers_col[i])),file=f)
            print(("\t\tnode [style=filled, color=white, penwidth={},"
                   "fillcolor={} shape=circle];".format(
                penwidth,
                layers_fill[i])),file=f)

            print(("\t\t"), end=' ',file=f)

            for a in range(layers[i]):
                print("l{}{} ".format(i + 1, a), end=' ',file=f)

            print(";",file=f)
            print(("\t\tlabel = {};".format(layers_str[i])),file=f)

            print("\t}\n",file=f)

        # Nodes
        for i in range(1, len(layers)):
            for a in range(layers[i - 1]):
                for b in range(layers[i]):
                    print("\tl{}{} -> l{}{}".format(i, a, i + 1, b),file=f)

        print("}",file=f)


    #dot.render('test-output/round-table.gv', view=True)


def buildXinput(data_folder):

    #########
    ##on recupere les donnees brutes

    #on recupere la conso
    conso_csv = os.path.join(data_folder, "conso_Y.csv")
    conso_df = pd.read_csv(conso_csv, sep=";", engine='c',header=0)  # engine en language C et position header pour accélérer le chargement
    conso_df['ds'] = pd.to_datetime(conso_df['date'] + " " + conso_df['time'])

    #on recupere les jours feries
    jours_feries_csv = os.path.join(data_folder, "joursFeries.csv")
    jours_feries_df = pd.read_csv(jours_feries_csv, sep=";")
    jours_feries_df.ds = pd.to_datetime(jours_feries_df.ds)

    #on recupere la meteo
    meteo_csv = os.path.join(data_folder, "meteoX_T.csv")
    meteo_df = pd.read_csv(meteo_csv, sep=";", engine='c', header=0)
    meteo_df['ds'] = pd.to_datetime(meteo_df['date'] + ' ' + meteo_df['time'])

    # on recupere les stations meteo
    stations_meteo_csv = os.path.join(data_folder, "StationsMeteoRTE.csv")
    stations_meteo_df = pd.read_csv(stations_meteo_csv, sep=";")


    ## Fin recup données brutes
    ###################

    #############
    #on reduit le problème a la maille horaire et à la conso france
    # on commence par ecarter les colonnes inutiles
    consoFrance_df = conso_df[['ds', 'Consommation NAT t0']]

    # et maintenant on ne garde que les heures pleines
    minutes = consoFrance_df['ds'].dt.minute
    indices_hours = np.where(minutes.values == 0.0)
    consoFranceHoraire_df = consoFrance_df.loc[indices_hours]
    consoFranceHoraire_df=consoFranceHoraire_df.reset_index(drop=True)


    # les index de ce sous-dataframe correspondent à celle du dataframe de base,
    # et donc sont pour l'instant des multiples de 4.
    # on va les réinitialiser pour avoir une dataframe "neuve"
    minutes = meteo_df['ds'].dt.minute
    mask = np.where(minutes.values == 0.0)
    meteoHoraire_df = meteo_df.loc[mask]
    meteoHoraire_df = meteoHoraire_df.reset_index(drop=True)

    meteo_obs_df = meteoHoraire_df[list(meteoHoraire_df.columns[meteoHoraire_df.columns.str.endswith("Th+0")]) + ['ds']]
    meteo_prev_df = meteoHoraire_df[list(meteoHoraire_df.columns[meteoHoraire_df.columns.str.endswith("Th+24")]) + ['ds']]
    ##################

    ##############
    ### on augmente nos variables
    consoFranceHoraire_df['month'] = consoFranceHoraire_df['ds'].dt.month
    # On isole aussi les heures
    consoFranceHoraire_df['hour'] = consoFranceHoraire_df['ds'].dt.hour
    consoFranceHoraire_df['dayOfWeek'] = consoFranceHoraire_df['ds'].dt.weekday

    # On sépare les jours de la semaine en week-end / pas week-end
    # De base, la fonction datetime.weekday() renvoie 0 => Lundi, 2 => Mardi, ..., 5 => Samedi, 6 => Dimanche
    # Ci-dessous, si on a un jour d ela semaine alors dans la colonne weekday on mettra 1, et 0 si c'est le week-end
    consoFranceHoraire_df['weekday'] = (consoFranceHoraire_df['ds'].dt.weekday < 5).astype(int)

    # on rajoute les lags
    consoFranceHoraire_df['lag1H'] = consoFranceHoraire_df['Consommation NAT t0'].shift(1)
    consoFranceHoraire_df['lag1D'] = consoFranceHoraire_df['Consommation NAT t0'].shift(24)
    consoFranceHoraire_df['lag1W'] = consoFranceHoraire_df['Consommation NAT t0'].shift(24 * 7)

    #On rajoute la temperature France
    colsToKeepWeather = [s for s in meteo_prev_df.columns.get_values() if 'Th+24' in s]
    meteo_prev_df['FranceTh+24'] = np.dot(meteo_prev_df[colsToKeepWeather], stations_meteo_df['Poids'])
    colsToKeepWeather = [s for s in meteo_obs_df.columns.get_values() if 'Th+0' in s]
    meteo_obs_df['FranceTh+0']=np.dot(meteo_obs_df[colsToKeepWeather], stations_meteo_df['Poids'])
    ###################


    #################
    #merge conso et meteo
    conso_meteo = pd.merge(consoFranceHoraire_df, meteo_obs_df, on='ds')
    conso_meteo = pd.merge(conso_meteo, meteo_prev_df, on='ds')
    #######

    ########
    ### On construit X et Y

    #On construit notre Xinput
    Xinput = conso_meteo.drop(['lag1H'], axis=1)  # variables calendaires, conso retardée, température prévue


    # shift a -24h de la prevision a 24h, puis ajout variable de lag J-1
    colsToKeepWeather = [s for s in Xinput.columns.get_values() if 'Th+24' in s]
    Xinput[colsToKeepWeather] = Xinput[colsToKeepWeather].shift(24)

    lag_colsToKeepWeather = [s + "_J_1" for s in colsToKeepWeather]
    Xinput[lag_colsToKeepWeather] = Xinput[colsToKeepWeather].shift(24)

    # shift de température réalisée à J-1
    colsToKeepWeather = [s for s in Xinput.columns.get_values() if 'Th+0' in s]
    lag_colsToKeepWeather = [s + "_J_1" for s in colsToKeepWeather]
    Xinput[lag_colsToKeepWeather] = Xinput[colsToKeepWeather].shift(24)


    ###########
    ###jointure des jours feries et rajout position année
    Xinput['dsDay'] = [datetime.date(d.year, d.month, d.day) for d in Xinput['ds']]
    jours_feries_df['dsDay'] = [datetime.date(d.year, d.month, d.day) for d in jours_feries_df['ds']]
    jours_feries_df = jours_feries_df.drop('ds', axis=1)

    # on peut joindre les tables maintenant
    Xinput = pd.merge(Xinput, jours_feries_df, how="left", on="dsDay")
    encodedHolidays = pd.get_dummies(Xinput[['holiday']], prefix="JF")
    encodedHolidays['JoursFeries'] = encodedHolidays.sum(axis=1)
    Xinput = pd.concat([Xinput, encodedHolidays], axis=1)

    Xinput = Xinput.drop(['holiday'], axis=1)
    Xinput = Xinput.drop(['dsDay'], axis=1)

    time = pd.to_datetime(Xinput['ds'], yearfirst=True)
    Xinput['posan'] = time.dt.dayofyear
    #################

    ################
    ##one hot encoding
    encodedDayOfWeek = pd.get_dummies(Xinput['dayOfWeek'], prefix="dayOfWeek")
    encodedMonth = pd.get_dummies(Xinput['month'], prefix="month")
    encodedHour = pd.get_dummies(Xinput['hour'], prefix="hour")

    Xinput = pd.concat([Xinput, encodedMonth, encodedDayOfWeek, encodedHour], axis=1)
    Xinput = Xinput.drop(['month', 'dayOfWeek', 'hour'],axis=1)
    ##############

    #On construit notre Y
    Yconso = Xinput[['ds', 'Consommation NAT t0']]
    Yconso.columns = ['ds', 'y']
    Xinput = Xinput.drop(['Consommation NAT t0'], axis=1)

    print(Xinput.columns.get_values())
    Xinput.to_csv(os.path.join(data_folder, "Xinput2.csv"))
    Yconso.to_csv(os.path.join(data_folder, "Yconso2.csv"))


def plot_load_timedelta(consoHoraire_df,var_load, year, month, day, delta_days):
    date_cible = datetime.datetime(year=year, month=month, day=day)
    date_lendemain_cible = date_cible + datetime.timedelta(days=delta_days)

    conso_periode = consoHoraire_df[(consoHoraire_df.ds >= date_cible)
                                      & (consoHoraire_df.ds <= date_lendemain_cible)]
    plt.plot(conso_periode['ds'], conso_periode[var_load], color='blue')
    plt.show()


def prepareDataSetEntrainementTest_TP1(Xinput, Yconso, dateDebut, dateRupture, nbJourlagRegresseur=0):
    dateStart = Xinput.iloc[0]['ds']

    DateStartWithLag = dateStart + pd.Timedelta(str(
        nbJourlagRegresseur) + ' days')  # si un a un regresseur avec du lag, il faut prendre en compte ce lag et commencer l'entrainement a la date de debut des donnees+ce lag
    XinputTest = Xinput[(Xinput.ds >= dateRupture)]

    XinputTrain = Xinput[(Xinput.ds < dateRupture) & (Xinput.ds > DateStartWithLag) & (Xinput.ds > dateDebut)]
    YconsoTrain = Yconso[(Yconso.ds < dateRupture) & (Yconso.ds > DateStartWithLag) & (Yconso.ds > dateDebut)]
    YconsoTest = Yconso[(Xinput.ds >= dateRupture)]

    return XinputTrain, XinputTest, YconsoTrain, YconsoTest

def selectDatesRandomly(XwithDs, quarter, nDays, seed=10):
    joursQuarter = np.unique(XwithDs.loc[np.where(XwithDs.ds.dt.quarter == quarter)].ds.dt.date)
    np.random.seed(seed)
    joursSelected = np.random.choice(joursQuarter, replace=False, size=nDays)
    return joursSelected

def prepareDataSetEntrainementTest_TP2(Xinput, Yconso, dateDebut, dateRupture, nbJourlagRegresseur=0, njoursEte=10,
                                   njoursHiver=10):
    dateStart = Xinput.iloc[0]['ds']

    # DateStartWithLag = dateStart + pd.Timedelta(str(nbJourlagRegresseur) + ' days')  # si un a un regresseur avec du lag, il faut prendre en compte ce lag et commencer l'entrainement a la date de debut des donnees+ce lag

    XinputTest1 = Xinput[(Xinput.ds >= dateRupture)]
    XinputTrain = Xinput[(Xinput.ds < dateRupture) & (Xinput.ds >= dateDebut)]  # & (Xinput.ds > DateStartWithLag)
    YconsoTrain = Yconso[(Yconso.ds < dateRupture) & (Xinput.ds >= dateDebut)]  # & (Yconso.ds > DateStartWithLag)
    YconsoTest1 = Yconso[(Yconso.ds >= dateRupture)]

    XinputTrain = XinputTrain.reset_index(drop=True)
    YconsoTrain = YconsoTrain.reset_index(drop=True)

    joursHiverSelectionne = selectDatesRandomly(XinputTrain, 4, njoursHiver, seed=10)
    print("Les jours d'hiver du jeu de test sont : ")
    print(joursHiverSelectionne)

    joursEteSelectionne = selectDatesRandomly(XinputTrain, 2, njoursEte, seed=20)
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

    XinputTrain = XinputTrain.drop(['ds'], axis=1)
    XinputTest1 = XinputTest1.drop(['ds'], axis=1)
    XinputTest2 = XinputTest2.drop(['ds'], axis=1)
    # indicesJoursEte=np.where(YconsoTrain.ds.dt.quarter==2)
    # joursHiver=
    # XinputTest2=Xinput[(Xinput.ds>=dateRupture)]

    # XinputTrain.reset_index(drop=True)
    # YconsoTrain.reset_index(drop=True)

    return XinputTrain, XinputTest1, XinputTest2, YconsoTrain, YconsoTest1, YconsoTest2

def modelError(Y, Yhat):
    Y = Y.reset_index(drop=True)

    relativeErrorsTest = np.abs((Y['y'] - Yhat) / Y['y'])
    errorMean = np.mean(relativeErrorsTest)
    errorMax = np.max(relativeErrorsTest)
    rmse = np.sqrt(mean_squared_error(Y['y'], Yhat))

    return relativeErrorsTest, errorMean, errorMax, rmse


def modelError2(YconsoReal, forecast, colName="Consommation NAT t0"):
    # attention cette frame est un subset d'une autre, il faut la reindexer depuis 0
    YconsoReal = YconsoReal.reset_index(drop=True)

    # TODO: calculer les erreurs relatives, moyenne et max avec numpy
    Yc = YconsoReal[colName].get_values().reshape(-1, 1)
    relativeErrorsTest = np.abs(Yc - forecast) / Yc
    errorMean = np.mean(relativeErrorsTest)
    errorMax = np.max(relativeErrorsTest)
    # END

    print("L'erreur relative moyenne est de : " + str(errorMean))
    print("L'erreur relative max est de : " + str(errorMax))

    return relativeErrorsTest, errorMean, errorMax

def evaluation_par(X, Y, Yhat, avecJF=True):
    Y['weekday'] = Y['ds'].dt.weekday
    Y['hour'] = Y['ds'].dt.hour
    if (avecJF):
        Y['JoursFeries'] = X['JoursFeries']
    Y['APE'] = np.abs(Y['y'] - Yhat) / Y['y']
    dataWD = Y[['weekday', 'APE']]
    groupedWD = dataWD.groupby(['weekday'], as_index=True)
    statsWD = groupedWD.aggregate([np.mean])
    dataHour = Y[['hour', 'APE']]
    groupedHour = dataHour.groupby(['hour'], as_index=True)
    statsHour = groupedHour.aggregate([np.mean])

    if (avecJF):
        dataJF = Y[['JoursFeries', 'APE']]
        groupedJF = dataJF.groupby(['JoursFeries'], as_index=True)
        statsJF = groupedJF.aggregate([np.mean])
    else:
        statsJF = None

    return statsWD, statsHour, statsJF


def evaluation(YTrain, YTest, YTrainHat, YTestHat):
    # Ytrain et Ytest ont deux colonnes : ds et y
    # YtrainHat et YTestHat sont des vecteurs
    ErreursTest, ErreurMoyenneTest, ErreurMaxTest, RMSETest = modelError(YTest, YTestHat)
    print("l'erreur relative moyenne de test est de:" + str(round(ErreurMoyenneTest * 100, 1)) + "%")
    print("l'erreur relative max de test est de:" + str(round(ErreurMaxTest * 100, 1)) + "%")
    print('le rmse de test est de:' + str(round(RMSETest, 0)))
    print()
    ErreursTest, ErreurMoyenneTest, ErreurMaxTest, RMSETest = modelError(YTrain, YTrainHat)
    print("l'erreur relative moyenne de train est de:" + str(round(ErreurMoyenneTest * 100, 1)) + "%")
    print("l'erreur relative max de train est de:" + str(round(ErreurMaxTest * 100, 1)) + "%")
    print('le rmse de test est de:' + str(round(RMSETest, 0)))



# #Xinput['dayOfWeek']=Xinput['ds'].dt.weekday
# #colsHolidays = [s for s in Xinput.columns.get_values() if ('JF' in s) or ('JoursFeries' in s)]
# #colsHolidays.append('JoursFeries')
# #Xinput=Xinput.drop(colsHolidays,axis=1)
# #encodedDayOfWeek = pd.get_dummies(Xinput['ds'].dt.weekday,prefix="dayOfWeek")
# #Xinput = pd.concat([Xinput, encodedDayOfWeek], axis=1)
# #Xinput.to_csv(os.path.join(data_folder, "Xinput.csv"))
# #encodedWeekDay = pd.get_dummies(Xinput['weekday'],prefix="weekday")
# #consoFranceHoraire_df['ds'].dt.hour
# #encodedDay = pd.get_dummies(Xinput['weekday'],prefix="weekday")
# #colsDWeekDay = [s for s in Xinput.columns.get_values() if 'weekday' in s]
# #Xinput=Xinput.drop(colsDWeekDay,axis=1)
# #Xinput['weekday']=(Xinput['ds'].dt.weekday < 5).astype(int)
# #Xinput.columns.get_values()
# #Xinput.to_csv(os.path.join(data_folder, "Xinput.csv"))
# jours_feries_csv = os.path.join(data_folder,"joursFeries.csv")
# jours_feries_df = pd.read_csv(jours_feries_csv, sep=";")
# jours_feries_df['ds']=pd.to_datetime(jours_feries_df['ds'])
# jours_feries_df['dsDay']=[datetime.date(d.year, d.month, d.day) for d in jours_feries_df['ds'] ]
# jours_feries_df=jours_feries_df.drop('ds',axis=1)
#
#
# #Xinput['dsDay']=[datetime.date(d.year, d.month, d.day) for d in Xinput['ds'] ]
# #Xinput['JoursFeries']=[any(datetime.date(jf.year, jf.month, jf.day)==datetime.date(d.year, d.month, d.day) for jf in jours_feries_df['dsDay']) for d in Xinput['dsDay']]
# #Xinput=pd.merge(Xinput, jours_feries_df, left_on='dsDay', right_index=True,how='left', sort=False);
# #Xinput=Xinput.join(jours_feries_df,on='dsDay')
# #Xinput= pd.merge(Xinput,jours_feries_df,how="left", on='dsDay')
# #Xinput=Xinput.drop('dsDay',axis=1)
# #Xinput[(Xinput.JoursFeries)]
# #encodedHolidays = pd.get_dummies(Xinput[['holiday']], prefix="JF")
# #encodedHolidays['JoursFeries'] = encodedHolidays.sum(axis=1)
# #Xinput = pd.concat([Xinput, encodedHolidays], axis=1)
# #Xinput = Xinput.drop(['holiday','dsDay'], axis=1)
# #Xinput.to_csv(os.path.join(data_folder, "Xinput.csv"))
# #Xinput
#
# # TODO: charger "meteoX_T.csv" dans "meteo_df"
# meteo_csv = os.path.join(data_folder, "meteoX_T.csv")
# meteo_df = pd.read_csv(meteo_csv, sep=";",engine='c',header=0)
# # END
#
# # TODO: créer une colonne "ds" dans "meteo_df" comme précédemment
# meteo_df['ds'] = pd.to_datetime(meteo_df['date'] + ' ' + meteo_df['time'])
# # END
#
# # TODO: afficher les 5 premières lignes de "meteo_df"
# meteo_df.head(5)
# # END
#
# minutes = meteo_df['ds'].dt.minute
# mask = np.where(minutes.values == 0.0)
# meteoHoraire_df = meteo_df.loc[mask]
# meteoHoraire_df = meteoHoraire_df.reset_index(drop=True)
# meteo_obs_df = meteoHoraire_df[list(meteoHoraire_df.columns[meteoHoraire_df.columns.str.endswith("Th+0")]) + ['ds']]
# meteo_obs_df.head(5)
#
# Xinput= pd.merge(Xinput,meteo_obs_df,how="left", on='ds')
#
# colsToKeepWeather=[s for s in Xinput.columns.get_values() if 'Th+0' in s]
# lag_colsToKeepWeather=[s+"_J_1" for s in colsToKeepWeather ]
# Xinput[lag_colsToKeepWeather]=Xinput[colsToKeepWeather].shift(24)
#
# Xinput.columns.get_values()
#
# Xinput.to_csv(os.path.join(data_folder, "Xinput.csv"))
#
# Xinput=Xinput[Xinput.ds>=jours_feries_df.iloc[10]['dsDay']]
# Xinput[['ds','JoursFeries','JF_11Novembre']]
# #Xinput