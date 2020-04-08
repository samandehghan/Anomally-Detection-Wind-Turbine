# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:24:24 2019

@author: dehgh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:37:27 2019

@author: dehgh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:29:56 2019

@author: dehgh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:41:51 2019

@author: dehgh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:11:30 2019

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
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
# load data
#file = 'Series5.xlsx'
#xl = pd.ExcelFile(file)
#dataset = xl.parse('Sheet1')
dataset = np.loadtxt('newinput.txt')
dataset=dataset.T
#dataset = newinputMain
#dataInput = pd.DataFrame(dataset)
#file = 'Series24.xlsx'
#xl = pd.ExcelFile(file)
#test = xl.parse('Sheet1')
test = np.loadtxt('test1_fault6mp10.txt', skiprows=1)
#traindata = np.loadtxt('traint.txt', skiprows=1)
## manually specify column names
#dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
#dataset.index.name = 'date'
## mark all NA values with 0
#dataset['pollution'].fillna(0, inplace=True)
## drop the first 24 hours
#dataset = dataset[24:]
## summarize first 5 rows
#print(dataset.head(5))
## save to file
#dataset.to_csv('pollution.csv')

#groups = [0, 1, 2, 3, 4]
#i = 1
## plot each column
#pyplot.figure()
#for group in groups:
#	pyplot.subplot(len(groups), 1, i)
#	pyplot.plot(values[:, group])
#	pyplot.title(dataset.columns[group], y=0.5, loc='right')
#	i += 1
#pyplot.show()



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
#dataset = read_csv('pollution.csv', header=0, index_col=0)
nLag = 1
values = dataset
# integer encode direction
# ensure all data is float
values = values.astype('float32')
# normalize features
reframed = series_to_supervised(values, nLag, 1)
#reframed.drop(reframed.columns[[0,9,10,13,14,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]], axis=1, inplace=True)
#a = list(range(4,reframed.shape[1]-2,5))
#reframed.drop(reframed.columns[[a]], axis=1, inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(reframed)
#scaled.drop(scaled.columns[4,5,7], axis=1, inplace=True)

train_X, train_y = scaled[:, 0:scaled.shape[1]-4], scaled[:, scaled.shape[1]-4:scaled.shape[1]]
#train_y = train_y[:-1]
# frame as supervised learning
# drop columns we don't want to predict
#print(reframed.head())


#values = reframed.values
#train_X, train_y = values[:, :-1], values[:, -1]
#Test dataset
#valuestest = test.values[:,1:5]

#valuestest = test.values
valuestest = test
#valuestest1 = pd.DataFrame(test)
# integer encode direction
# ensure all data is float
valuestest = valuestest.astype('float32')
reframedtest = series_to_supervised(valuestest, nLag, 1)
reframedtest.drop(reframedtest.columns[[0,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]],axis=1, inplace=True)
reframedtest1 = reframedtest.reindex(['var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var9(t-1)', 'var12(t-1)', 'var5(t-1)', 'var6(t-1)', 'var7(t-1)', 'var8(t-1)', 'var2(t)', 'var3(t)', 'var4(t)', 'var9(t)', 'var12(t)', 'var5(t)', 'var6(t)', 'var7(t)', 'var8(t)'], axis=1)
# normalize features
#valuestest = valuestest1
#reframedtest = series_to_supervised(valuestest, nLag, 1)
#reframedtest.drop(reframedtest.columns[[0,9,10,13,14,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]], axis=1, inplace=True)
#a = list(range(4,reframedtest.shape[1]-2,5))
#reframedtest.drop(reframedtest.columns[[a]], axis=1, inplace=True)
scalertest = MinMaxScaler(feature_range=(0, 1))
scaledtest = scalertest.fit_transform(reframedtest1)
#scaledtest.drop(scaledtest.columns[4,5,7], axis=1, inplace=True)
test_X, test_y = scaledtest[:, 0:scaledtest.shape[1]-4], scaledtest[:, scaledtest.shape[1]-4:scaledtest.shape[1]]

#test_X, test_y = scaledtest[:, :-1], scaledtest[:, -1]
#valuestest = test.values
## integer encode direction
## ensure all data is float
##valuestest = valuestest.astype('float32')
## normalize features
#scalertest = MinMaxScaler(feature_range=(0, 1))
#scaledtest = scalertest.fit_transform(valuestest)
#test_X_be, test_y = scaledtest[:, :-1], scaledtest[:, -1]
#
#test_y = test_y[:-1]
#
## frame as supervised learning
#test_X = series_to_supervised(test_X_be, 1, 1)
# drop columns we don't want to predict
#reframedtest.drop(reframedtest.columns[[5,6,7,8]], axis=1, inplace=True)
#print(reframedtest.head())


#valuestest = scaledtest
#test_X, test_y = scaledtest[:, :-1], scaledtest[:, -1]

#train_X = train_X.reshape((train_X.shape[0], nLag, train_X.shape[1]))
#train_y = train_y.reshape((train_y.shape[0], train_y.shape[1]))
#train_y = train_y.reshape((train_y.shape[0],1, train_y.shape[1]))
#
#test_X = test_X.reshape((test_X.shape[0], nLag, test_X.shape[1]))
#test_y = test_y.reshape((test_y.shape[0], test_y.shape[1]))
#test_y = test_y.reshape((test_y.shape[0],1, test_y.shape[1]))
#
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#LSTM Fit Model

#model = Sequential()
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(1))
#model.compile(loss='mae', optimizer='adam')
## fit network
#history = model.fit(train_X, train_y, epochs=50, batch_size=5, verbose=2, shuffle=False)
#plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
#train_X.shape[1]


#Main Alg
#model = Sequential()
#model.add(LSTM(250, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(RepeatVector(1))
#model.add(LSTM(250, activation='relu', return_sequences=True))
#model.add(TimeDistributed(Dense(4, activation='relu', W_constraint=nonneg())))
##model.add(TimeDistributed(Dense(2)))
#model.compile(loss='mse', optimizer='Adagrad')
#history = model.fit(train_X, train_y, epochs=10, batch_size=1000, verbose=2)

model = LinearRegression()
model.fit(train_X, train_y)



 #make a prediction
yhat = model.predict(test_X)
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]*(nLag)))
### invert scaling for forecast
##test_X_be = test_X_be[:-1,:]
#yhat = yhat.reshape(yhat.shape[0],yhat.shape[2])

inv_yhat = np.column_stack((test_X,yhat))
inv_yhat = scalertest.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,[14,15,16,17]]

#inv_yhat = inv_yhat[:, scaledtest.shape[1]-4:scaledtest.shape[1]]
### invert scaling for actual
#test_y = test_y.reshape(test_y.shape[0],test_y.shape[2])
inv_y = np.column_stack((test_X,test_y))
inv_y = scalertest.inverse_transform(inv_y)
inv_y = inv_y[:, [14,15,16,17]]

#inv_y = inv_y[:, scaledtest.shape[1]-4:scaledtest.shape[1]]
### calculate RMSE
rmse0 = sqrt(mean_squared_error(inv_y[:,0], inv_yhat[:,0]))
rmse1 = sqrt(mean_squared_error(inv_y[:,1], inv_yhat[:,1]))
rmse2 = sqrt(mean_squared_error(inv_y[:,2], inv_yhat[:,2]))
rmse3 = sqrt(mean_squared_error(inv_y[:,3], inv_yhat[:,3]))
print('Test RMSE: %.3f' % rmse0)
#
                #fig1 = pyplot.figure(figsize=(16,9))
#axes = pyplot.gca()
#axes.set_xlim([10,720])
#axes.set_ylim([0,80])
#pyplot.plot(inv_y[:,0], label='original')
#pyplot.plot(inv_yhat[:,0], label='estimated')
#pyplot.ylabel('VFD Value')
#pyplot.xlabel('Time(Minutes)')
#pyplot.legend()
#pyplot.savefig("vfd39.png")
#pyplot.show()

#pyplot.plot(inv_y[:,1], label='original')
#pyplot.plot(inv_yhat[:,1], label='estimated')
#pyplot.ylabel('Flow Rate Value')
#pyplot.xlabel('Time(Minutes)')
##pyplot.savefig("fr18.png")
#pyplot.legend()
#pyplot.show()


#fig2 = pyplot.figure(figsize=(16,9))
#axes = pyplot.gca()
#axes.set_xlim([10,720])
##axes.set_ylim([0,100000])
#pyplot.plot(inv_y[:,1], label='original')
#pyplot.plot(inv_yhat[:,1], label='estimated')
#pyplot.ylabel('Pressure Sensor Value')
#pyplot.xlabel('Time(Minutes)')
#pyplot.legend()
#plt.title('Dataset 36')
#pyplot.savefig("rp36.eps")
#
#pyplot.show()

#pyplot.plot(inv_y[:,3], label='original')
#pyplot.plot(inv_yhat[:,3], label='estimated')
#pyplot.ylabel('Inlet Pressure Sensor Value')
#pyplot.xlabel('Time(Minutes)')
#pyplot.savefig("ip39.png")
#pyplot.legend()
#pyplot.show()
#errS39 = np.square(inv_y[0:718,1] - inv_yhat[1:719,1]) 
############ Predicion Error
err0 = inv_y[:,0] - inv_yhat[:,0]
err0 = err0.reshape(-1,1)
err0norm = scaler.fit_transform(err0) 
err1 = inv_y[:,1] - inv_yhat[:,1]
err1 = err1.reshape(-1,1)
err1norm = scaler.fit_transform(err1) 
err2 = inv_y[:,2] - inv_yhat[:,2]
err2 = err2.reshape(-1,1)
err2norm = scaler.fit_transform(err2) 
err3 = inv_y[:,3] - inv_yhat[:,3]
err3 = err3.reshape(-1,1)
err3norm = scaler.fit_transform(err3) 
errtot100 = np.row_stack((err0norm,err1norm,err2norm,err3norm))

############ Squared Predicion Error

errs0 = np.square(inv_y[:,0] - inv_yhat[:,0])
errs0 = errs0.reshape(-1,1)
errs0norm = scaler.fit_transform(errs0) 
errs1 = np.square(inv_y[:,1] - inv_yhat[:,1])
errs1 = errs1.reshape(-1,1)
errs1norm = scaler.fit_transform(errs1) 
errs2 = np.square(inv_y[:,2] - inv_yhat[:,2])
errs2 = errs2.reshape(-1,1)
errs2norm = scaler.fit_transform(errs2) 
errs3 = np.square(inv_y[:,3] - inv_yhat[:,3])
errs3 = errs3.reshape(-1,1)
errs3norm = scaler.fit_transform(errs3) 
errtots100 = np.row_stack((errs0norm,errs1norm,errs2norm,errs3norm))

#fig3 = pyplot.figure(figsize=(16,9))
#axes = pyplot.gca()
#axes.set_xlim([10,720])
#axes.set_ylim([0,100000])
#pyplot.ylabel('Pressure Sensor Error')
#pyplot.xlabel('Time(Minutes)')
#pyplot.plot(err1, label='SPE')
#pyplot.legend()
##pyplot.savefig("erpresure38.png")
#
#pyplot.show()


#err2 = np.square(inv_y[0:718,0] - inv_yhat[1:719,0]) 
#fig4 = pyplot.figure(figsize=(16,9))
#axes = pyplot.gca()
#axes.set_xlim([10,720])
##axes.set_ylim([0,100000])
#pyplot.ylabel('VFD Error')
#pyplot.xlabel('Time(Minutes)')
#pyplot.plot(err2, label='SPE')
#pyplot.legend()
#pyplot.savefig("ervfd39.png")
#
#pyplot.show()

#for i in range(4):
#    rmse1.append(sqrt(mean_squared_error(inv_y[:,i], inv_yhat[:,i])))
#
          #fig5 = pyplot.figure(figsize=(16,9))
#axes = pyplot.gca()
#axes.set_xlim([0,720])
#axes.set_ylim([0,1.1])
#pyplot.plot(test_X[:,0], label='VFD')
#pyplot.plot(test_X[:,1], label='Flow')
#pyplot.plot(test_X[:,2], label='Rise Presure')
#pyplot.plot(test_X[:,3], label='Inelet Presure')
#pyplot.ylabel('Parameters')
#pyplot.xlabel('Time(Minutes)')
#pyplot.legend()
#pyplot.savefig("all39.png")
#
          #pyplot.show()
#Forecasting the number of occupant
#def forecast(model, batch_size, row):
#	X = row.values[:,-1]
#	X = X.reshape(1, 1, len(X))
#	yhat = model.predict(X, batch_size=batch_size)
#	return yhat[0,0]
#
#yforecast = forecast(model,100,test_y)
#print(rmse0)
#print(rmse1)


#Bar Chart
#N = 39
#fig6 = pyplot.figure(figsize=(16,9))
#
#errvf = ( 3.7977 ,2.7478, 2.0808 ,1.6365 ,2.7542, 2.5880, 4.0602,2.6094, 4.2785, 1.6913,1.6833 ,2.3503,1.8310,2.2722,1.8115,2.2491,4.1046,1.6606,0 ,1.4370,4.0718,2.8438,3.4525,1.2353,2.0706 ,1.1408 ,2.2774 ,3.0467,4.0910 ,1.6702 ,4.0445 ,3.3984 ,1.8456 ,3.2073 ,3.2463 ,4.4143 ,2.8311 ,4.7442)
#errp = (34.7813 ,8.6780,11.2105 ,7.0516 ,23.0261 ,8.2765 ,23.7191 ,35.6923,12.1524  ,7.7175  ,7.2314  ,13.3528 ,16.0547 ,15.6083  ,13.1233  ,23.6704  ,31.6095    ,8.3583 ,17.9092,239.7865 ,62.8154,150.7345,106.8056 ,76.9913 ,38.2001,135.5616,111.3038,167.4092,120.2488,226.2134,145.8388,112.7069,186.963 ,123.5395,246.1635 ,81.1299 ,278.1884)
#ind = np.arange(N)    # the x locations for the groups
#width = .35       # the width of the bars: can also be len(x) sequence
#p1 = plt.bar(ind, errp, width)
#p2 = plt.bar(ind, errvf, width)
#plt.ylabel('Prediction Error')
#plt.xlabel('Dataset Number')
#
#plt.title('Prediction Error for Different Dataset')
#plt.xticks(ind, ('0','1', '2', '3', '4', '5' ,'6','7','8', '9', '10', '11', '12' ,'13','14','15', '16', '17', '18', '19' ,'20','21','22', '23', '24', '25', '26' ,'27','28','29','30','31', '32', '33', '34', '35' ,'36'))
##plt.yticks(np.arange(0, 81, 10))
#plt.legend((p1[0], p2[0]), ('Pressure Error', 'VFD Error'))
#pyplot.savefig("Barchart.png")
#
#plt.show()

#ErrorM = np.row_stack((err2,err3,err4,err5,err6,err7,err8,err9,err10,err11,err12,err13,err14,err15,err16,err17,err18,err19,err20,err21,err22,err23,err24,err25,err26,err27,err28,err29,err30,err31,err32,err33,err34,err35,err36,err37,err38,err39))
#ErrorSM = np.row_stack((errS2,errS3,errS4,errS5,errS6,errS7,errS8,errS9,errS10,errS11,errS12,errS13,errS14,errS15,errS16,errS17,errS18,errS19,errS20,errS21,errS22,errS23,errS24,errS25,errS26,errS27,errS28,errS29,errS30,errS31,errS32,errS33,errS34,errS35,errS36,errS37,errS38,errS39))
#dtwscore = dtw(errS2,errS3)         
#ds2 = dtw.distance_matrix_fast(ErrorSM)
#
#Z = linkage(ErrorM, 'ward')
#plt.figure(figsize=(25, 15))
#plt.title('Hierarchical Clustering Dendrogram')
#dendrogram(Z)
#pyplot.savefig("Dendrogram.png")
#plt.show()

#fig1 = pyplot.figure(figsize=(5,2))
#axes = pyplot.gca()
#axes.set_xlim([1,720])
###axes.set_ylim([0,100000])
#pyplot.xlabel('Time(Minutes)')
#pyplot.plot(values[:,0], label='VFD')
#pyplot.legend()
#pyplot.savefig("in0.eps")
#
##
#pyplot.show()
#
#fig2 = pyplot.figure(figsize=(5,2))
#axes = pyplot.gca()
#axes.set_xlim([1,720])
###axes.set_ylim([0,100000])
#pyplot.xlabel('Time(Minutes)')
#pyplot.plot(values[:,1], label='Flow')
#pyplot.legend()
#pyplot.savefig("in1.eps")
#
##
#pyplot.show()
#
#fig3 = pyplot.figure(figsize=(5,2))
#axes = pyplot.gca()
#axes.set_xlim([1,720])
###axes.set_ylim([0,100000])
#pyplot.xlabel('Time(Minutes)')
#pyplot.plot(values[:,2], label='Outlet Pressure')
#pyplot.legend()
#pyplot.savefig("in2.eps")
#
##
#pyplot.show()
#
#fig4 = pyplot.figure(figsize=(5,2))
#axes = pyplot.gca()
#axes.set_xlim([1,720])
###axes.set_ylim([0,100000])
#pyplot.xlabel('Time(Minutes)')
#pyplot.plot(values[:,3], label='Inlet Pressure')
#pyplot.legend()
##
#pyplot.savefig("in3.eps")
#pyplot.show()




#def fancy_dendrogram(*args, **kwargs):
#    max_d = kwargs.pop('max_d', None)
#    if max_d and 'color_threshold' not in kwargs:
#        kwargs['color_threshold'] = max_d
#    annotate_above = kwargs.pop('annotate_above', 0)
#
#    ddata = dendrogram(*args, **kwargs)
#
#    if not kwargs.get('no_plot', False):
#        plt.title('Hierarchical Clustering Dendrogram (truncated)')
#        plt.xlabel('sample index or (cluster size)')
#        plt.ylabel('distance')
#        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
#            x = 0.5 * sum(i[1:3])
#            y = d[1]
#            if y > annotate_above:
#                plt.plot(x, y, 'o', c=c)
#                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
#                             textcoords='offset points',
#                             va='top', ha='center')
#        if max_d:
#            plt.axhline(y=max_d, c='k')
#    return ddata
##
#
##
##plt.figure(figsize=(25, 15))
##plt.title('Hierarchical Clustering Dendrogram')
##fancy_dendrogram(Z,show_contracted=True,annotate_above=40,)
##
##plt.savefig("DendrogramWind.png")
##plt.show()
##          
#          
#          plt.figure(figsize=(25, 15))
#plt.title('Hierarchical Clustering Dendrogram')
#fancy_dendrogram(Z,show_contracted=True,annotate_above=40,)
#
#plt.savefig("DendrogramWind.png")
#plt.show()
#          
#          
#          
##ErrorM = np.row_stack((errtot2,errtot3,errtot4,errtot5,errtot6,errtot7,errtot8,errtot9,errtot10,errtot11,errtot12,errtot13,errtot14,errtot15,errtot16,errtot17,errtot18))
##
#ErrorM = np.row_stack((errtot2.T,errtot3.T,errtot4.T,errtot5.T,errtot6.T,errtot7.T,errtot8.T,errtot9.T,errtot10.T,errtot11.T,errtot12.T,errtot13.T,errtot14.T,errtot15.T,errtot16.T,errtot17.T,errtot18.T))
##
#Z = linkage(ErrorM, 'ward')

## EWMA main Code

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
ewma = pd.Series.ewm

#
diffy= inv_yhat[:,1] - inv_y[:,1]
y = diffy
df = pd.Series(y)
# take EWMA in both directions then average them
fwd = ewma(df,span=20).mean() # take EWMA in fwd direction
bwd = ewma(df[::-1],span=20).mean() # take EWMA in bwd direction
filtered = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
filtered = np.mean(filtered, axis=0 ) # average
filteredmax = filtered  + 4.5*np.mean(diffy) 
filteredmin = filtered  - 4.5*np.mean(diffy) 
plt.figure(figsize=(25, 15))
plt.title('filtered and raw data')
plt.plot(y, color = 'orange')
plt.plot(filtered, color='green')
plt.plot(filteredmax, color='red')
plt.plot(filteredmin, color='blue')
plt.xlabel('samples')
plt.ylabel('amplitude')
plt.savefig("test7nofault.eps")
plt.show()
##
#newinput = np.row_stack((filtered1,filtered2,filtered3,filtered4,filtered5,filtered6,filtered7,filtered8,filtered9))

## Heatmap
#import pandas as pd
#import seaborn as sns
#
#def heatMap(df):
#    #Create Correlation df
#    corr = df.corr()
#    #Plot figsize
#    fig, ax = plt.subplots(figsize=(10, 10))
#    #Generate Color Map
#    colormap = sns.diverging_palette(220, 10, as_cmap=True)
#    #Generate Heat Map, allow annotations and place floats in map
#    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#    #Apply xticks
#    plt.xticks(range(len(corr.columns)), corr.columns);
#    #Apply yticks
#    plt.yticks(range(len(corr.columns)), corr.columns)
#    #show plot
#    plt.show()