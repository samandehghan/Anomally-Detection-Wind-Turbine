# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:52:51 2019

@author: dehgh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:46:36 2019

@author: dehgh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:04:02 2019

@author: Sam Dehghan
"""

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
from numpy import array
from numpy import hstack

#import matplotlib.pyplot as plt
# load data
df = pd.read_csv('traint.txt',delimiter="\t")
#df = tftrain.groupby(np.arange(len(tftrain))//10).mean()
# Dropping features are not gonna use for training

Input_features=['v1','Beta_r','Omega_r','Tau_r','Beta_1_m1','Beta_1_m2','Beta_2_m1','Beta_2_m2','Beta_3_m1','Beta_3_m2','beta_1_f','beta_2_f','beta_3_f']
# Defining the target variable for prediction
target_features =  ['Beta']
#target_features =  ['Tau_r','Beta','Beta1','Beta3']

num_feature = len(Input_features)
num_feature_out = len(target_features)
# Defining the number of steps we want to predict into the future
time_lag = 1
# Create a new data-frame with the time-shifted data. 
## make sure that the sign is negative to predict the future not the past
df_targets = df[target_features].shift(-time_lag)
df_input = df[Input_features]
# Create Input and Output signal of the plant and remove the last rows of the signal including NaN value due to the shifting the data-frame
pl_input = df_input.values[0:-time_lag]
pl_output = df_targets.values[:-time_lag]
pl_output = pl_output.reshape((pl_output.shape[0], num_feature_out))
#Number of the observation after eliminating NaN rows
num_obs = len(pl_input)

# Import test data
dftest = pd.read_csv('test1_fault6mp10.txt',delimiter="\t")
#dftest = dftests.groupby(np.arange(len(dftests))//10).mean()

# Dropping features are not gonna use for training

Input_features_test=['v1','Beta_r','Omega_r','Tau_r','Beta_1_m1','Beta_1_m2','Beta_2_m1','Beta_2_m2','Beta_3_m1','Beta_3_m2','beta_1_f','beta_2_f','beta_3_f']
# Defining the target variable for prediction
target_features_test =  ['Beta']
#target_features_test =  ['Tau_r','Beta','Beta1','Beta3']

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

num_train = train_X_scaled.shape[0]
num_input = pl_input.shape[1]
num_output = pl_output.shape[1]



def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
n_steps = 10
dtrain = hstack((train_X_scaled,train_y_scaled))
train_X_scaled,train_y_scaled= split_sequences(dtrain,n_steps)

dtest = hstack((test_X_scaled,test_y_scaled))
test_X_scaled,test_y_scaled = split_sequences(dtest,n_steps)
##



# Algorithm parameters
Num_hid1 = 8
Num_hid2 = 8
Num_hid3 = 4
active_hid1 = 'relu'
active_hid2 = 'relu'
active_hid3 = 'relu'
loss_fun = 'mean_squared_error'
optimizer_type = 'adam'
num_epoch = 10
sizeOfBatch = 10
#Main Alg
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(n_steps, num_feature)))
model.add(Dense(5, activation=active_hid2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(train_X_scaled, train_y_scaled, epochs=num_epoch, batch_size=sizeOfBatch, verbose=2)

#model.fit(train_X_scaled, train_y_scaled, epochs=20, verbose=2)

##model = Sequential()
##model.add(LSTM(16, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
##model.add(RepeatVector(1))
##model.add(LSTM(16, activation='relu', return_sequences=True))
##model.add(TimeDistributed(Dense(7*nLag-28, activation='tanh', W_constraint=nonneg())))
##model.add(TimeDistributed(Dense(2, activation='relu')))
#
##model.add(TimeDistributed(Dense(2)))
#

 #make a prediction
#yhat = model.predict(test_X_scaled)
yhat = model.predict(test_X_scaled)
#ypred = scaler_target.inverse_transform(yhat)
ypred1 = scaler_target.inverse_transform(yhat)

ytrue = test_y
width_in_inches = 26.66
height_in_inches = 15
dots_per_inch = 57

fig1 = plt.figure(figsize=(width_in_inches,height_in_inches))
plt.plot(ytrue[8:], label='Original')
plt.plot(ypred1, label='Estimated')
plt.savefig("tempOsma02.eps")
plt.legend()
plt.show()

