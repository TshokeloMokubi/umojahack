import pandas as pd
import numpy as np
import pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import cluster
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn import pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
%matplotlib inline


train_df = pd.read_csv('/Train.csv').set_index('ID')
sample_set = pd.read_csv('/SampleSubmission.csv').set_index('ID')
testing = pd.read_csv('/Test.csv').set_index('ID')
weather_df = pd.read_csv('/Weather.csv')

train_df['Timestamp'] = pd.to_datetime(train_df.Timestamp)
train_df['Year'] = train_df.Timestamp.dt.year
train_df['month'] = train_df.Timestamp.dt.month
train_df['day'] = train_df.Timestamp.dt.day
train_df['hour'] = train_df.Timestamp.dt.hour
train_df['minute'] = train_df.Timestamp.dt.minute
train_df['second'] = train_df.Timestamp.dt.second

X = train_df.drop(labels = ['Timestamp', 'ETA','Year'], axis = 'columns')
y = train_df['ETA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state=8)

from sklearn import svm
clf = svm.SVC()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
predictions = pd.DataFrame(y_pred, columns = ['ETA'])
predictions.to_csv('sub.csv', index = False)





