#selecting data files from the folder using glob

import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

lr = LinearRegression()
ls = Lasso()
en = ElasticNet() 
dtr = DecisionTreeRegressor()
knr = KNeighborsRegressor()
gbr = GradientBoostingRegressor(n_estimators = 21)
models = [lr, ls,en, dtr, knr, gbr]

files = glob.glob("studentdata.csv")

df = pd.read_csv(files[0])

# exploring dataset
print(df.columns)
print(df.head())

#null or NaN values
print(df.isnull().sum())

domain = df['MHRDName'].unique()
print(len(domain))

# seperating 'Bachelor of Science (Honours) (Botany)' from the dataset
required_df = df.loc[df['MHRDName'] == 'Bachelor of Science (Honours) (Botany)']
print(required_df.shape)

#creating new column(clas) for classification of ETT and ETP
required_df ['clas'] = np.where(required_df['ETT_100']>0.0, 1, 0)
print(required_df.head())

features = ['CA_100', 'MTT_50', 'ETT_100',
       'ETP_100', 'Course_Att', 'CA_1', 'CA_2', 'CA_3', 'CA_4','clas']
required_df = required_df[features]
print(required_df.head())

#Handling missing values in the dataset
required_df.fillna(-1, inplace=True)
#classification of ETT and ETP using random forest classifier


#from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

target = ['clas', 'ETT_100', 'ETP_100']
features = ['CA_100', 'MTT_50', 'Course_Att', 'CA_1', 'CA_2', 'CA_3', 'CA_4',]

x_train, x_test, y_train, y_test = train_test_split(required_df[features], required_df[target], test_size=0.2)

classifier=RandomForestClassifier() 
classifier=classifier.fit(x_train,y_train['clas'])
predicted=classifier.predict(x_test)
print(y_test, predicted)
print('classification accuracy :', accuracy_score(predicted, y_test['clas']))
#training and testing for ETT
features = ['CA_100', 'MTT_50', 'Course_Att', 'CA_1', 'CA_2', 'CA_3', 'CA_4',]
prev = 0
X_train, X_test, Y_train, Y_test = train_test_split(required_df[features], required_df[['ETT_100']], test_size=0.2)
for model in models:
    model.fit(X_train, Y_train)
    p = model.predict(X_test)
    if prev < accuracy_score(Y_test, p.round()):
        prev = accuracy_score(Y_test, p.round())
        ett_model = model

print(ett_model)

#training and testing for ETP

prev = 0
X_train, X_test, Y_train, Y_test = train_test_split(required_df[features], required_df[['ETP_100']], test_size=0.2)
for model in models:
    model.fit(X_train, Y_train)
    p = model.predict(X_test)
    print(accuracy_score(Y_test, p.round()))
    if prev < accuracy_score(Y_test, p.round()):
        prev = accuracy_score(Y_test, p.round())
        etp_model = model

print(etp_model)


### if user want to check the performance of the student
marks_ca = list(map(int, input('if no etp enter -1 for it: ').strip().split('\t')))

clas = classifier.predict([marks_ca])
if clas == 1:
    pred = ett_model.predict([marks_ca])
else:
    pred = etp_model.predict([marks_cal])

print(pred)
    
