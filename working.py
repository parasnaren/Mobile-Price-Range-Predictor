import numpy as np # linear algra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv').drop('id', axis=1)

train_col= list(train.columns)
test_col= list(test.columns)

temp = train[['four_g','three_g']].copy()
tmp = train[['battery_power','mobile_wt','px_height','px_width','ram','price_range']].copy()

corrmat = train.corr()
sns.heatmap(corrmat, square=True)

corrmat = tmp.corr()
sns.heatmap(corrmat, square=True)

var = 'ram'
sns.boxplot(x='price_range', y=var, data=train)
plt.scatter(train['ram'],train['price_range'])

var = 'clock_speed'
sns.boxplot(x='price_range', y=var, data=train)

var = 'n_cores'
sns.boxplot(x='price_range', y=var, data=train)

train['cpu'] = train['n_cores']*train['clock_speed']
var = 'cpu'
sns.boxplot(x='price_range', y=var, data=train)

var = 'battery_power'
sns.boxplot(x='price_range', y=var, data=train)

train['temp'] = train['battery_power']*train['ram']
var = 'temp'
sns.boxplot(x='price_range', y=var, data=train)

train['ram'] = train['ram']#**2
train['px'] = train['px_height']*train['px_width']

######## Splitting the data into train, test and validation
y = train['price_range']
X = train.drop(['price_range','cpu','temp','px'], axis=1)

from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.15, random_state=1)
y_train.value_counts()
y_test.value_counts()

X_train2, X_val, y_train2, y_val = tts(X_train, y_train, test_size=0.15, random_state=1)
y_train2.value_counts()
y_val.value_counts()

####### Random Forest Regression #########
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

pred1 = pd.Series(regressor.predict(X_test))
pred11 = pd.Series(regressor.predict(X_val))

y_pred = pd.Series(regressor.predict(X_test)).apply(lambda x: round(x))

c = 0
for i, j in zip(y_pred, y_test):
    if i == j:
        c+=1
print("Forest Accuracy = ", (c*100)/y_test.shape[0])

###### Gradient Boosting Regression #######

reg = GBR()
reg.fit(X_train, y_train)

pred2 = pd.Series(reg.predict(X_test))
pred22 = pd.Series(reg.predict(X_val))

y_pred2 = pd.Series(reg.predict(X_test)).apply(lambda x: round(x))

d = 0
for i, j in zip(y_pred2, y_test):
    if i == j:
        d+=1
print("GBR Accuracy = ", (d*100)/y_test.shape[0])

###### Stacking by predicting on X_test ########
a = pd.DataFrame(np.column_stack((pred1,pred2)))

regg = RandomForestRegressor(n_estimators = 10, random_state = 0)
regg.fit(a, y_test)

# Stacking by predicting on X_val
b = pd.DataFrame(np.column_stack((pred11,pred22)))

y_predfinal = pd.Series(regg.predict(b)).apply(lambda x: round(x))

f = 0
for i, j in zip(y_predfinal, y_val):
    if i == j:
        f+=1
print("Final Accuracy = ", (f*100)/y_val.shape[0])

######## Prediction on test data ##########
test = pd.read_csv('test.csv').drop('id', axis=1)
X_train, X_val, y_train, y_val = tts(X, y, test_size=0.15, random_state=1)

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

pred1 = pd.Series(regressor.predict(test))
pred11 = pd.Series(regressor.predict(X_val))

###### Gradient Boosting Regression #######
reg = GBR()
reg.fit(X_train, y_train)

pred2 = pd.Series(reg.predict(test))
pred22 = pd.Series(reg.predict(X_val))

###### Stacking by predicting on X_test ########
b = pd.DataFrame(np.column_stack((pred11,pred22)))
a = pd.DataFrame(np.column_stack((pred1,pred2)))

regg = RandomForestRegressor(n_estimators = 10, random_state = 0)
regg.fit(b, y_test)

# Stacking by predicting on X_val
prediction = pd.Series(regg.predict(a)).apply(lambda x: round(x))

t = pd.read_csv('sample.csv')
t['price_range'] = prediction
t.to_csv('submission_1.csv', index=False)

pd.Series(t['price_range']).value_counts()
