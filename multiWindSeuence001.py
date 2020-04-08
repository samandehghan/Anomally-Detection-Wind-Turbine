# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:30:46 2020

@author: dehgh
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
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import hstack

#import matplotlib.pyplot as plt
# load data
df = pd.read_csv('traint.txt',delimiter="\t")
Input_features=['v1','Beta_1_m1','Beta_1_m2','Beta_2_m1','Beta_2_m2','Beta_3_m1','Beta_3_m2','beta_1_f','beta_2_f','beta_3_f','Omega_g_m1','Omega_g_m2','Omega_r_m1','Omega_r_m2','P_g_m','tau_g_m']
df = df[Input_features]
train_value = df.values
scaler_train = MinMaxScaler()
train = scaler_train.fit_transform(train_value)

df_input = df[Input_features]

# Import test data
dftest = pd.read_csv('fault4_ver00.txt',delimiter="\t")
Input_features_test=['v1','Beta_1_m1','Beta_1_m2','Beta_2_m1','Beta_2_m2','Beta_3_m1','Beta_3_m2','beta_1_f','beta_2_f','beta_3_f','Omega_g_m1','Omega_g_m2','Omega_r_m1','Omega_r_m2','P_g_m','tau_g_m']
dftest = dftest[Input_features_test]
#dftest = dftests.groupby(np.arange(len(dftests))//10).mean()
test_value = dftest.values
scaler_test = MinMaxScaler()
test = scaler_test.fit_transform(test_value)


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
n_steps_in, n_steps_out = 20, 1

train_X,train_y= split_sequences(train,n_steps_in, n_steps_out)

test_X,test_y = split_sequences(test,n_steps_in, n_steps_out)
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
num_epoch = 1
sizeOfBatch = 10
num_feature = train_value.shape[1]
#Main Alg
#model = Sequential()
#model.add(LSTM(20, activation='relu', input_shape=(n_steps_in, num_feature)))
#model.add(RepeatVector(n_steps_out))
#model.add(LSTM(20, activation='relu', return_sequences=True))
#model.add(TimeDistributed(Dense(num_feature)))
#model.compile(optimizer='adam', loss='mse')
#model.fit(train_X, train_y, epochs=num_epoch, batch_size=sizeOfBatch, verbose=2)


#

 #make a prediction
yhat = model.predict(test_X)
yhat = yhat.reshape((yhat.shape[0], yhat.shape[2]))
ypred1 = scaler_test.inverse_transform(yhat)

ytrue = dftest.values
width_in_inches = 26.66
height_in_inches = 15
dots_per_inch = 57

#fig1 = plt.figure(figsize=(width_in_inches,height_in_inches))
#plt.plot(ytrue[175000:225000,1], label='Original')
#plt.plot(ypred1[175000:225000,1], label='Estimated')
#plt.savefig("fault3.eps")
#plt.legend()
#plt.show()




errnorm = np.square(ytrue[20:,]-ypred1)
err = np.square(ytrue[20:,]-ypred1)
plt.plot(err[:,10])
plt.plot(threshold[10,:]+5)
plt.savefig("fault4.eps")
plt.legend()
plt.show()


#xer_train = test[20:,]
#model = Sequential()
#model.add(Dense(16, input_dim=16, activation='relu'))
#model.add(Dense(16, activation='sigmoid'))
## compile the keras model
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
## fit the keras model on the dataset
#model.fit(xer_train, err, epochs=10, batch_size=100)
#accuracy = model.evaluate(xer_train, err, verbose=0)

##
#import matplotlib.pyplot as plt
#import pandas as pd
#import numpy as np
#ewma = pd.Series.ewm
#
##
#diffy= errnorm[:,15]
#y = diffy
#df = pd.Series(y)
## take EWMA in both directions then average them
#fwd = ewma(df,span=500).mean() # take EWMA in fwd direction
#bwd = ewma(df[::-1],span=500).mean() # take EWMA in bwd direction
#filtered = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
#filtered = np.mean(filtered, axis=0 ) # average
#filteredmax = filtered  + 4.5*np.mean(diffy) 
#filteredmin = filtered  - 4.5*np.mean(diffy) 
#plt.figure(figsize=(25, 15))
#plt.title('filtered and raw data')
#plt.plot(y, color = 'orange')
#plt.plot(filtered, color='green')
#plt.plot(filteredmax, color='red')
#plt.plot(filteredmin, color='blue')
#plt.xlabel('samples')
#plt.ylabel('amplitude')
##plt.savefig("test7nofault.eps")
#plt.show()
#
#fil15 = filtered
#fil15max = filteredmax
#fil15min = filteredmin
##
##clData = np.row_stack((cl0,cl1,cl2,cl3,cl4,cl5,cl6,cl7,cl8,cl9,cl10,cl11,cl12,cl13,cl14,cl15))
#threshold = np.row_stack((fil0,fil1,fil2,fil3,fil4,fil5,fil6,fil7,fil8,fil9,fil10,fil11,fil12,fil13,fil14,fil15))
#upper_thresh = np.row_stack((fil0max,fil1max,fil2max,fil3max,fil4max,fil5max,fil6max,fil7max,fil8max,fil9max,fil10max,fil11max,fil12max,fil13max,fil14max,fil15max))
#lower_thresh = np.row_stack((fil0min,fil1min,fil2min,fil3min,fil4min,fil5min,fil6min,fil7min,fil8min,fil9min,fil10min,fil11min,fil12min,fil13min,fil14min,fil15min))