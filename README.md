# Notions du Projet ML-Model-for-S-P-500-prediction

Ce projet vise à développer des modèles simples de machine learning pour prédire la valeur de clôture de l'indice S&P500. L'approche repose sur des notions de base en data science, telles que :

- **Collecte et Prétraitement des données**  
  Les données financières sont récupérées via la librairie **yfinance**. Ensuite, plusieurs indicateurs techniques sont calculés pour enrichir le jeu de données.

- **Extraction d'indicateurs techniques**  
  - **SMA20 (Simple Moving Average sur 20 jours)** : Cet indicateur lisse les variations du prix et permet de dégager une tendance à court terme.  
  - **RSI (Relative Strength Index)** : Un oscillateur qui mesure la vitesse et la variation des mouvements de prix, aidant à identifier des situations de surachat ou de survente.  
  - **MACD (Moving Average Convergence Divergence)** : (Optionnel dans certaines versions) Cet indicateur combine deux moyennes mobiles pour détecter les changements de momentum et la convergence/divergence des tendances.

- **Séparation des données**  
  L'ensemble des données est divisé en jeux d'entraînement et de test à l'aide de `train_test_split` afin d'évaluer les performances du modèle.

- **Modélisation**  
  Deux approches principales sont mises en œuvre pour la prédiction :

  ## 1. Régression Linéaire (`linear-RegV2.py`)
  
  - **Objectif** : Utiliser un modèle linéaire pour établir une relation simple entre les features et la valeur de clôture du S&P500.
  - **Features utilisées** :  
    - *Days* : Le nombre de jours écoulés depuis le début de la période.
    - *SMA20* : La moyenne mobile sur 20 jours.
    - *RSI* : Le Relative Strength Index.
    - (Optionnellement, on peut ajouter le MACD, pour mesurer le momentum.)
  - **Évaluation** : La performance du modèle est mesurée en calculant l'erreur quadratique moyenne racine (**RMSE**).

  ## 2. Random Forest (`randomForest.py`)
  
  - **Objectif** : Utiliser un modèle d'ensemble (RandomForestRegressor) qui combine plusieurs arbres de décision pour capturer des relations non linéaires dans les données.
  - **Features utilisées** :  
    - De façon similaire au modèle linéaire, des features telles que *Days*, *SMA20*, *RSI* et potentiellement *MACD* sont utilisées pour l'entraînement.
  - **Évaluation** : Le modèle est évalué par le calcul du RMSE afin de mesurer la précision des estimations.

Ces deux modèles sont conçus pour permettre une comparaison entre une approche simple (régression linéaire) et une approche plus robuste (Random Forest) en vue de déterminer laquelle fournit de meilleures prédictions pour les mouvements de l'indice S&P500.

N'hésitez pas à expérimenter en modifiant les features ou en ajustant les paramètres des modèles afin d'améliorer leurs performances.
