from pandas import read_csv
from datetime import datetime
#from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.constraints import nonneg
import numpy as np
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt
# load data
df = pd.read_csv('traint.txt',delimiter="\t")

# Dropping features are not gonna use for training

Input_features=['v1','Beta_r','Omega_r','Tau_r','Beta','Beta1','Beta3']
# Defining the target variable for prediction
target_features =  ['Tau_r','Beta','Beta1','Beta3']
num_feature = len(target_features)
# Defining the number of steps we want to predict into the future
time_lag = 20
# Create a new data-frame with the time-shifted data. 
## make sure that the sign is negative to predict the future not the past
df_targets = df[target_features].shift(-time_lag)
df_input = df[Input_features]
# Create Input and Output signal of the plant and remove the last rows of the signal including NaN value due to the shifting the data-frame
pl_input = df_input.values[0:-time_lag]
pl_output = df_targets.values[:-time_lag]
pl_output = pl_output.reshape((pl_output.shape[0], num_feature))
#Number of the observation after eliminating NaN rows
num_obs = len(pl_input)

# Import test data
dftest = pd.read_csv('traint.txt',delimiter="\t")

# Dropping features are not gonna use for training

Input_features_test=['v1','Beta_r','Omega_r','Tau_r','Beta','Beta1','Beta3']
# Defining the target variable for prediction
target_features_test =  ['Tau_r','Beta','Beta1','Beta3']
num_feature_test = len(target_features_test)
# Defining the number of steps we want to predict into the future
# Create a new data-frame with the time-shifted data. 
## make sure that the sign is negative to predict the future not the past
dftest_targets = dftest[target_features_test]
dftest_input = dftest[Input_features_test]
# Create Input and Output signal of the plant and remove the last rows of the signal including NaN value due to the shifting the data-frame
pl_input_test = dftest_input.values[0:-time_lag]
pl_output_test = dftest_targets.values[:-time_lag]
pl_output_test = pl_output_test.reshape((pl_output_test.shape[0], num_feature_test))
#Number of the observation after eliminating NaN rows
num_obs_test = len(pl_input_test)
#percentage of the observation using for training

# create train and test dataset
train_X = pl_input
test_X = pl_input_test

train_y = pl_output
test_y = pl_output_test

# Number of the input and ouuput signal
num_input = pl_input.shape[1]
num_output = pl_output.shape[1]

# Scale data
scaler_input = MinMaxScaler()
train_X_scaled = scaler_input.fit_transform(train_X)
scaler_input_test = MinMaxScaler()

test_X_scaled = scaler_input_test.fit_transform(test_X)

scaler_target = MinMaxScaler()
train_y_scaled = scaler_target.fit_transform(train_y)
scaler_target_test = MinMaxScaler()

test_y_scaled = scaler_target_test.fit_transform(test_y)

train_X_scaled = train_X_scaled.reshape((train_X_scaled.shape[0], 1, train_X_scaled.shape[1]))
test_X_scaled = test_X_scaled.reshape((test_X_scaled.shape[0], 1, test_X_scaled.shape[1]))


Num_hid1 = 16
Num_hid2 = 8
Num_hid3 = 4
active_hid1 = 'relu'
active_hid2 = 'relu'
active_hid3 = 'relu'
loss_fun = 'mean_squared_error'
optimizer_type = 'adam'

#Main Alg
#model = Sequential()
#model.add(LSTM(Num_hid1, activation=active_hid1,input_shape=(train_X_scaled.shape[1], train_X_scaled.shape[2])))
#model.add(Dense(num_feature))
#model.compile(loss=loss_fun, optimizer=optimizer_type)
#
#
#model.fit(train_X_scaled, train_y_scaled, epochs=10, batch_size=100, verbose=2)
#

 #make a prediction
yhat = model.predict(test_X_scaled)
ypred = scaler_target_test.inverse_transform(yhat)

ytrue = test_y

fig1 = plt.figure(figsize=(16,9))
plt.plot(ytrue[:,0], label='original')
plt.plot(ypred[:,0], label='estimated')
plt.ylabel('Blade 1 Angel')
plt.xlabel('Time(ms)')
#plt.savefig("LSTMver06b1.eps")
plt.legend()
plt.show()


#import matplotlib.pyplot as plt
#import pandas as pd
#import numpy as np
#ewma = pd.Series.ewm
#
##
#diffy= ypred[:,1]
#y = diffy
#df = pd.Series(y)
## take EWMA in both directions then average them
#fwd = ewma(df,span=500).mean() # take EWMA in fwd direction
#bwd = ewma(df[::-1],span=500).mean() # take EWMA in bwd direction
#filtered = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
#filtered = np.mean(filtered, axis=0 ) # average
#filteredmax = filtered  + np.mean(diffy) 
#filteredmin = filtered  -np.mean(diffy) 
#plt.figure(figsize=(25, 15))
#plt.title('filtered and raw data')
#plt.plot(y, color = 'orange')
#plt.plot(filtered, color='green')
##plt.plot(filteredmax, color='red')
##plt.plot(filteredmin, color='blue')
#plt.xlabel('samples')
#plt.ylabel('amplitude')
#plt.savefig("test7nofault.eps")
#plt.show()