Estimation des valeures de précipataions à l'aide des entrées météorlogiques.
les données issues de la direction générale de lamétéorlogie pour la ville de benguerir_2018/
les données ne sont pas affichés ici car ils sont confidentielles.

une préparation des donéées :feature engineering, feature extraction, standarization.

créer  deux modéles de prévision : XGBoost et LSTM
les résultats montrant que le modele LSTM a fait des réslatats mieux que XGBoost cela due au fait que LTSM est faite pour les series chronologiques.
 les resultats de XGBoost:
Train Mean Absolute Error: 0.03630926
Train Root Mean Squared Error: 0.14740486
Test Mean Absolute Error: 0.06273403
Test Root Mean Squared Error: 0.23027587

 les resultats de LSTM:
 
 Train Mean Absolute Error: 0.04289090192340187
Train Root Mean Squared Error: 0.2091397059930701
Test Mean Absolute Error: 0.044815656019529686
Test Root Mean Squared Error: 0.15769988306404653
