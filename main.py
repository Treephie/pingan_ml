# coding: utf-8
import pandas as pd


header = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE', 'Y']

# load data
data = pd.read_csv('PINGAN-2018-train_demo.csv', sep=',')
print(data.shape)


# pre process. nan


# train test split
train_X = data[['TIME', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE']][:60000]
train_Y = data[['Y']][:60000]
test_X = data[['TIME', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE']][60001:69001]
test_Y = data[['Y']][60001:69001]


# fit and predict
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()
clf.fit(train_X, train_Y)
pred = clf.predict(test_X)

print('actual:\n', test_Y)
print('pred:\n', pred)
