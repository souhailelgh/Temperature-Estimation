import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
from sklearn.preprocessing import LabelBinarizer , LabelEncoder,OneHotEncoder 
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.model_selection import StratifiedKFold # pour balancer dataset 
from sklearn.impute import SimpleImputer
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
import xgboost as xgb

data = pd.read_csv('C:\\Users\\hp\\Desktop\\deep\\Benguerir.CSV', engine='python')
data['Date']=pd.to_datetime(data['Date']) 
print(data.columns)

data.fillna(method='pad', inplace= True)
data.fillna(method='bfill',inplace= True)

data['mois']= data['Date'].dt.to_period('M') 
data['jour']= data['Date'].dt.to_period('D') 
data['heure']= data['Date'].dt.to_period('H')

data['Tmax']=data.groupby('heure')['T'].transform('max')
data['Tmin']=data.groupby('heure')['T'].transform('min')
data['Tdmax']=data.groupby('heure')['Td'].transform('max')
data['Tdmin']=data.groupby('heure')['Td'].transform('min')

data['Humidite']=data.groupby('heure')['Humidite'].transform('mean')
data['Precipitation']=data.groupby('heure')['Precipitation'].transform('sum')
data['Pression']=data.groupby('heure')['Pression'].transform('mean')
data['Radiation_Global']=data.groupby('heure')['Radiation_Global'].transform('mean')
data['Td']=data.groupby('heure')['Td'].transform('mean')
data['T']=data.groupby('heure')['T'].transform('mean')
data['Vapeur_Press']=data.groupby('heure')['Vapeur_Press'].transform('mean')
data['Vent_Vitesse']=data.groupby('heure')['Vent_Vitesse'].transform('min')
data['Vent_Dir']=data.groupby('heure')['Vent_Dir'].transform('min')

data= data.drop_duplicates(subset=['heure'] )
print(data.info())


def windcode(x):
    if x< 25:
        return 1
    if x< 50:
        return 2
    if x< 75:
        return 3
    if x< 95:
        return 4
    if x< 120:
        return 5
    if x< 145:
        return 6
    if x< 170:
        return 7
    if x< 195:
        return 8
    if x< 220:
        return 9
    if x< 245:
        return 10
    if x< 270:
        return 11
    if x< 295:
        return 12
    if x< 320:
        return 13
    if x< 345:
        return 14
    if x< 360:
        return 15

data['Vent_Dir'].fillna(method='pad', inplace= True)
data['Vent_Dir'].fillna(method='bfill',inplace= True)



print(data.info())

data['Vent_Dir']= data['Vent_Dir'].apply(lambda x : windcode(x))


print(data['Vent_Dir'])
print(type(data['Vent_Dir']))


data['Pression_diff1']= data['Pression'].diff( periods = -1)
data['Pression_diff2']= data['Pression'].diff( periods = -2)
data['Pression_diff3']= data['Pression'].diff( periods = -3)

data['Humidite_diff1']= data['Humidite'].diff( periods = -1)
data['Humidite_diff2']= data['Humidite'].diff( periods = -2)
data['Humidite_diff3']= data['Humidite'].diff( periods = -3)

data['Vent_Dir_diff1']= data['Vent_Dir'].diff( periods = -1)


data= data.drop('Date', axis=1)
data= data.drop('mois', axis=1)
data= data.drop('jour', axis=1)
data= data.drop('heure', axis=1)

print(data.info())

target= data.Precipitation.values
inputs  = data.drop('Precipitation', axis=1)


#dataset = data.Precipitation.values #numpy.ndarray
dataset= target
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
y_train, y_test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

dataset= inputs
dataset= dataset.values
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, inputs.shape[1]))
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
X_train, X_test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print('X_train is :',X_train)
print('X_test is :',X_test)
print('y_train is :',y_train)
print('y_test is :',y_test)

plt.plot(y_test)
plt.title('test')
plt.show()

plt.plot(y_train)
plt.title('train')
plt.show()


model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('X_train est: \n',X_train)
print('X_test est: \n',X_train)


train_predict = model.predict(X_train).reshape(-1,1)
test_predict = model.predict(X_test).reshape(-1,1)
# invert predictions
#train_predict = scaler.inverse_transform(train_predict)
#Y_train = scaler.inverse_transform([Y_train])
#test_predict = scaler.inverse_transform(test_predict)
#Y_test = scaler.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(y_train, train_predict))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train, train_predict)))
print('Test Mean Absolute Error:', mean_absolute_error(y_test, test_predict))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test, test_predict)))



plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()



plt.figure(figsize=(8,4))
plt.plot( Y_test.reshape(-1,1), marker='.', label="actual")
plt.plot(  test_predict.reshape(-1,1), 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
