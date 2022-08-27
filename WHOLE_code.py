##########################################
######### 21-2 GLM Final Project #########
##########################################

######## 1. EDA
import math
import scipy as sp
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
sns.set()

import warnings

df = pd.read_csv("~\\data1.csv", header=0,engine='python')  
df.info()
df = df.loc[0:117]
df = df[["날짜","정책","정책지수","누적정책지수","서울","용산구","광진구","서대문구","영등포구","서초구","소비자물가지수","한국은행기준금리","원달러환율","가계대출","서울시부동산심리지수","서울실거래지수","도심권실거래지수","동북권실거래지수","서북권실거래지수","서남권실거래지수","동남권실거래지수"]]
df.columns = ["date","1","PI","PI2","Seoul","Yongsan-gu","Gwangjin-gu","Seodaemun-gu","Yeongdeungpo-gu","Seocho-gu","CPI","BR","ER","HL","EI","SI","CI","NEI","NWI","SWI","SEI"]
df.tail()

sns.set(rc={'figure.figsize':(16,10)})
sns.boxplot(data=df.iloc[:,1:12])

feature = df[["PI","CPI","BR","ER","HL","EI"]]

whole1 = df[["Seoul","PI","CPI","BR","ER","HL","EI"]]

whole2 = df[["Yongsan-gu","PI","CPI","BR","ER","HL","EI"]]
whole3 = df[["Gwangjin-gu","PI","CPI","BR","ER","HL","EI"]]
whole4 = df[["Seodaemun-gu","PI","CPI","BR","ER","HL","EI"]]
whole5 = df[["Yeongdeungpo-gu","PI","CPI","BR","ER","HL","EI"]]
whole6 = df[["Seocho-gu","PI","CPI","BR","ER","HL","EI"]]

label = df[["Seoul","Yongsan-gu","Gwangjin-gu","Seodaemun-gu","Yeongdeungpo-gu","Seocho-gu"]]

label1 = df[["Seoul"]]
label2 = df[["Yongsan-gu"]]
label3 = df[["Gwangjin-gu"]]
label4 = df[["Seodaemun-gu"]]
label5 = df[["Yeongdeungpo-gu"]]
label6 = df[["Seocho-gu"]]

date=df[["date"]]

sns.set(rc={'figure.figsize':(16,10)})
sns.boxplot(data=label)

plt.figure(figsize=(16,10))
label.plot()
plt.title("Price of Apartment in Seoul(Won)")

plt.figure(figsize=(16,10))
feature.PI.plot()
plt.title("Policy Index")

plt.figure(figsize=(16,10))
feature.CPI.plot()
plt.title("Customer Price Index")

plt.figure(figsize=(16,10))
feature.BR.plot()
plt.title("Base Rate by Bank of Korea")

plt.figure(figsize=(16,10))
feature.ER.plot()
plt.title("Exchange Rate")

plt.figure(figsize=(16,10))
feature.HL.plot()
plt.title("Housing Loan")

plt.figure(figsize=(16,10))
feature.EI.plot()
plt.title("Real-estate Index")

import pandas_profiling

#pr=df.profile_report() 
#df.profile_report()
#pr.to_file('/Users/kimhyunmin/data.html')

corr = df.corr()
corr1 = feature.corr()
corr2 = whole2.corr()
corr3 = whole3.corr()
corr4 = whole4.corr()
corr5 = whole5.corr()
corr6 = whole6.corr()

sns.clustermap(corr, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1, )
sns.clustermap(corr1, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1, )
sns.clustermap(corr2, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1, )
sns.clustermap(corr3, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1, )
sns.clustermap(corr4, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1, )
sns.clustermap(corr5, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1, )
sns.clustermap(corr6, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1, )

######## 2-1. LSTM ########
# load libraries
import math
import pandas as pd
import numpy as np
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM,Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# load the dataset
df = pd.read_csv("~\\data1.csv", header=0,engine='python') 
df = df.loc[0:117]
df = df[["날짜","정책","정책지수","누적정책지수","서울","용산구","광진구","서대문구","영등포구","서초구","소비자물가지수","한국은행기준금리","원달러환율","가계대출","서울시부동산심리지수","서울실거래지수","도심권실거래지수","동북권실거래지수","서북권실거래지수","서남권실거래지수","동남권실거래지수"]]
df.columns = ["date","1","PI","PI2","Seoul","Yongsan-gu","Gwangjin-gu","Seodaemun-gu","Yeongdeungpo-gu","Seocho-gu","CPI","BR","ER","HL","EI","SI","CI","NEI","NWI","SWI","SEI"]
df.tail()

feature = df[["PI","CPI","BR","ER","HL","EI"]]

whole1 = df[["Seoul","PI","CPI","BR","ER","HL","EI"]]

whole2 = df[["Yongsan-gu","PI","CPI","BR","ER","HL","EI"]]
whole3 = df[["Gwangjin-gu","PI","CPI","BR","ER","HL","EI"]]
whole4 = df[["Seodaemun-gu","PI","CPI","BR","ER","HL","EI"]]
whole5 = df[["Yeongdeungpo-gu","PI","CPI","BR","ER","HL","EI"]]
whole6 = df[["Seocho-gu","CPI","PI","BR","ER","HL","EI"]]

label = df[["Seoul","Yongsan-gu","Gwangjin-gu","Seodaemun-gu","Yeongdeungpo-gu","Seocho-gu"]]

label1 = df[["Seoul"]]
label2 = df[["Yongsan-gu"]]
label3 = df[["Gwangjin-gu"]]
label4 = df[["Seodaemun-gu"]]
label5 = df[["Yeongdeungpo-gu"]]
label6 = df[["Seocho-gu"]]

# normalize the dataset
scaler = MinMaxScaler()
xdata = scaler.fit_transform(feature)
ydata = scaler.fit_transform(label2)
wholedata=scaler.fit_transform(whole2)

# Window -> X timestep back
step_back = 1
X_whole, Y_whole = [], []
for i in range(len(wholedata)-step_back - 1):
    a = wholedata[i:(i+step_back), 1:]
    X_whole.append(a)
    Y_whole.append(wholedata[i + step_back, 0])
X_whole = np.array(X_whole); Y_whole = np.array(Y_whole)
X_whole.shape

# CV
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import keras.backend as K 
import tensorflow as tf

kfold=KFold(10)

K.clear_session()

cvscores = []

for train, test in kfold.split(X_whole, Y_whole):

    X_train, X_test, y_train, y_test = train_test_split(X_whole,Y_whole, test_size = 0.1)
    X_train = np.reshape(X_train, (X_train.shape[0], step_back, 6))
    X_test = np.reshape(X_test, (X_test.shape[0], step_back, 6))
    
    output_size=7 #feature+1
    activ_func="relu"
    dropout=0.25
    loss="mean_squared_error" 
    
    model = Sequential()
    model.add(LSTM(100,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(150))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer='adam',metrics =["mse"])

    model.fit(X_train, y_train, epochs=50, batch_size=24, verbose=0,shuffle=False)
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# split into train and test sets
train_size = int(len(xdata)*0.9)+4
x_train_dataset, x_test_dataset = xdata[0:train_size,:], xdata[train_size:,:]
y_train_dataset, y_test_dataset = ydata[0:train_size], ydata[train_size:]

# Window -> X timestep back
step_back= 1
X_train, Y_train = [], []
for i in range(len(x_train_dataset)-step_back - 1):
    a = x_train_dataset[i:(i+step_back), :]
    X_train.append(a)
    Y_train.append(y_train_dataset[i + step_back, 0])
X_train = np.array(X_train); Y_train = np.array(Y_train)
    
X_test, Y_test = [], []
for i in range(len(x_test_dataset)-step_back - 1):
    a = x_test_dataset[i:(i+step_back), :]
    X_test.append(a)
    Y_test.append(y_test_dataset[i + step_back, 0])
X_test = np.array(X_test); Y_test = np.array(Y_test)

print(X_train.shape); print(Y_train.shape);             
print(X_test.shape); print(Y_test.shape)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], step_back, 6))
X_test  = np.reshape(X_test, (X_test.shape[0], step_back, 6))
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test  = np.reshape(Y_test, (Y_test.shape[0], 1))

# setup a LSTM network in keras
output_size=7 #feature+1
activ_func="relu"
dropout=0.25
loss="mean_squared_error" 

model_lstm = Sequential()
model_lstm.add(LSTM(100,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dropout(dropout))
model_lstm.add(Dense(150))
model_lstm.add(Dense(units=output_size))
model_lstm.add(Activation(activ_func))

model_lstm.compile(loss=loss, optimizer='adam')
model_lstm.summary()
lstm_history = model_lstm.fit(X_train, Y_train, epochs=50, batch_size=24, validation_data=(X_test, Y_test), shuffle=False, verbose=1)

# Estimate model performance
trainScore = model_lstm.evaluate(X_train, Y_train, verbose=1)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

# Evaluate the skill of the Trained model
testPredict  = model_lstm.predict(X_test)
testPredict
testPredict.shape

result1=[]
for i in range(0,testPredict.shape[0]):
    result1.append(testPredict[i][0])
    i=+1

result1

# invert predictions
testPredict1 = scaler.inverse_transform(result1)

result_pred=[]
for i in range(0,testPredict1.shape[0]):
    result_pred.append(testPredict1[i][0])
    i=+1

result_pred

result11=[]
for i in range(0,len(result1)):
    result11.append(result1[i][0])
    i=+1
result11=np.reshape(result11,(-1,1))

#rmse
math.sqrt(sum((result11-Y_test)**2)/6)

#plt.figure(figsize=(16,9))
plt.scatter(result11,Y_test)

# plot baseline and predictions
#plt.figure(figsize=(16,9))
plt.plot(result_pred)
plt.plot(label2[-6:].values)
plt.show()

######## 2-2. CNN-LSTM ########
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from tqdm import tqdm
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

from sklearn.preprocessing import MinMaxScaler

# settings

label=label2

#feature.sort_index(ascending=False).reset_index(drop=True)
#label.sort_index(ascending=False).reset_index(drop=True)

feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

feature = feature_scaler.fit_transform(feature)
feature = pd.DataFrame(feature)

label = label_scaler.fit_transform(label)
label = pd.DataFrame(label)

def make_dataset(df, label, window_size):
    feature_list = []
    label_list = []
    for i in range(len(df) - window_size):
        feature_list.append(np.array(df.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

import os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


# CV
from sklearn.model_selection import train_test_split,KFold,cross_val_score

kfold=KFold(10)
cvscores = []

import random
seed = 1234
random.seed(seed)

from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,LSTM,Dropout,Conv1D, MaxPooling1D, TimeDistributed

for train, test in kfold.split(feature,label):
    
    train_feature, test_feature = feature[0:106], feature[93:106]
    train_label, test_label = label[0:106], label[93:106]
    pred_feature, pred_label = feature[105:118], label[105:118]

    train_feature, train_label = make_dataset(train_feature,train_label, 6)
    test_feature, test_label = make_dataset(test_feature, test_label, 6)
    pred_feature, pred_label = make_dataset(pred_feature, pred_label, 6)
    
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
    
    subsequences = 2
    timesteps = x_train.shape[1]//subsequences
    X_train_series_sub = x_train.reshape((x_train.shape[0], subsequences, timesteps, 6)) 
    X_valid_series_sub = x_valid.reshape((x_valid.shape[0], subsequences, timesteps, 6))
    test_feature_series_sub = test_feature.reshape((test_feature.shape[0], subsequences, timesteps, 6))
    pred_feature_series_sub = pred_feature.reshape((pred_feature.shape[0], subsequences, timesteps, 6))
    
    output_size=6 
    neurons=100 
    loss="mean_squared_error" 
    
    model_cnn_lstm = Sequential()
    model_cnn_lstm.add(TimeDistributed(Conv1D(filters=128, kernel_size=2, 
                                          activation='relu'),
                                   input_shape=(None, X_train_series_sub.shape[2],
                                                X_train_series_sub.shape[3])))
    model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model_cnn_lstm.add(TimeDistributed(Dropout((0.5))))
    model_cnn_lstm.add(TimeDistributed(Flatten()))
    model_cnn_lstm.add(LSTM(100, activation='relu'))
    model_cnn_lstm.add(Dropout(0.25))
    model_cnn_lstm.add(Dense(150))
    model_cnn_lstm.add(Dense(1))
    model_cnn_lstm.compile(loss=loss, optimizer="adam", metrics=["mse"])
    model_cnn_lstm.fit(X_train_series_sub, y_train,
                    epochs=50, 
                    verbose=1, 
                    validation_data=(X_valid_series_sub, y_valid), 
                    callbacks=[early_stop, checkpoint])
    scores = model_cnn_lstm.evaluate(test_feature_series_sub, test_label, verbose=0)
    cvscores = (scores * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))        

# prediction
test_pred = model_cnn_lstm.predict(test_feature_series_sub)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(test_label, test_pred)) 

test_pred = label_scaler.inverse_transform(test_pred)
test_label = label_scaler.inverse_transform(test_label)

pred = model_cnn_lstm.predict(pred_feature_series_sub)
true = pred_label

RMSE(true, pred)

pred = label_scaler.inverse_transform(pred)
pred

true = label2.tail(6)
true = np.array(true)
true

plt.plot(true, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()

######## 2-3. GRU ########
# load libraries
import math
import pandas as pd
import numpy as np
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# normalize the dataset
scaler = MinMaxScaler()
xdata = scaler.fit_transform(feature)
ydata = scaler.fit_transform(label2)
wholedata=scaler.fit_transform(whole2)

# Window -> X timestep back
step_back = 1
X_whole, Y_whole = [], []
for i in range(len(wholedata)-step_back - 1):
    a = wholedata[i:(i+step_back), 1:]
    X_whole.append(a)
    Y_whole.append(wholedata[i + step_back, 0])
X_whole = np.array(X_whole); Y_whole = np.array(Y_whole)
X_whole.shape

# CV
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import keras.backend as K 
import tensorflow as tf

kfold=KFold(10)

K.clear_session()

cvscores = []

for train, test in kfold.split(X_whole, Y_whole):

    X_train, X_test, y_train, y_test = train_test_split(X_whole, Y_whole, test_size = 0.1)
    X_train = np.reshape(X_train, (X_train.shape[0], step_back, 6))
    X_test  = np.reshape(X_test, (X_test.shape[0], step_back, 6))
    
    output_size=7
    activ_func="relu"
    dropout=0.25
    loss="mean_squared_error" 
    
    model = Sequential()
    model.add(GRU(75, return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(10))
    model.add(Dropout(dropout))
    model.add(GRU(units=30, return_sequences=True))
    model.add(Dense(units=output_size,activation=activ_func))

    model.compile(loss=loss, optimizer='adam',metrics =["mse"])

    model.fit(X_train, y_train, epochs=50, batch_size=24, verbose=0,shuffle=False)
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# split into train and test sets
train_size = int(len(xdata)*0.9)+4
x_train_dataset, x_test_dataset = xdata[0:train_size,:], xdata[train_size:,:]
y_train_dataset, y_test_dataset = ydata[0:train_size], ydata[train_size:]

# Window -> X timestep back
step_back= 1
X_train, Y_train = [], []
for i in range(len(x_train_dataset)-step_back - 1):
    a = x_train_dataset[i:(i+step_back), :]
    X_train.append(a)
    Y_train.append(y_train_dataset[i + step_back, 0])
X_train = np.array(X_train); Y_train = np.array(Y_train)
    
X_test, Y_test = [], []
for i in range(len(x_test_dataset)-step_back - 1):
    a = x_test_dataset[i:(i+step_back), :]
    X_test.append(a)
    Y_test.append(y_test_dataset[i + step_back, 0])
X_test = np.array(X_test); Y_test = np.array(Y_test)

print(X_train.shape); print(Y_train.shape);             
print(X_test.shape); print(Y_test.shape)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], step_back, 6))
X_test  = np.reshape(X_test, (X_test.shape[0], step_back, 6))
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test  = np.reshape(Y_test, (Y_test.shape[0], 1))

# setup a GRU network in keras
output_size=7
activ_func="relu"
dropout=0.25
loss="mean_squared_error" 

model_gru = Sequential()
model_gru.add(GRU(75, return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
model_gru.add(Dense(10))
model_gru.add(Dropout(0.25))
model_gru.add(GRU(units=30, return_sequences=True))
model_gru.add(Dense(units=output_size, activation="relu"))

model_gru.compile(loss='mean_squared_error', optimizer='adam')
model_gru.summary()

gru_history = model_gru.fit(X_train, Y_train, epochs=50, batch_size=24, validation_data=(X_test, Y_test), shuffle=False, verbose=1)

# Estimate model performance
trainScore = model_gru.evaluate(X_train, Y_train, verbose=1)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

# Evaluate the skill of the Trained model
testPredict  = model_gru.predict(X_test)
testPredict.shape

result1=[]
for i in range(0,testPredict.shape[0]):
    result1.append(testPredict[i][0])
    i=+1
result1

# invert predictions
testPredict1 = scaler.inverse_transform(result1)

result_pred=[]
for i in range(0,testPredict1.shape[0]):
    result_pred.append(testPredict1[i][0])
    i=+1

result_pred

result11=[]
for i in range(0,len(result1)):
    result11.append(result1[i][0])
    i=+1
result11=np.reshape(result11,(-1,1))

#rmse
math.sqrt(sum((result11-Y_test)**2)/6)

#plt.figure(figsize=(10,6))
plt.scatter(result11,Y_test)

# plot baseline and predictions
#plt.figure(figsize=(10,6))
plt.plot(result_pred)
plt.plot(label2[-6:].values)
plt.show()