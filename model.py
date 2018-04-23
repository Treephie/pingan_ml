# coding: utf-8
import pandas as pd


def init_model(path_train):
    header = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE', 'Y']
    # load data
    data = pd.read_csv(path_train, sep=',')
    # print(data.shape)
    # train test split
    # train_X = data[['TIME', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE']]
    train_X = data[['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE']]
    train_Y = data[['Y']]

    # fit
    from sklearn.tree import DecisionTreeRegressor
    clf = DecisionTreeRegressor()
    clf.fit(train_X, train_Y)

    return clf


if __name__ == '__main__':
    init_model()
