import numpy as np
import pandas as pd
df=pd.read_csv("studentdata.csv")

print(df.shape)
print(df.head)
print(df.describe)
features=df.columns[:]
print(features)
features=df[features]
print(features.shape)
c=features['MHRDName']
print(c)
print(c.nunique)
print(c.unique)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
m=le.fit_transform(c.values)
print(m)
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values=np.nan, strategy="mean")
imp=imp.fit_transform(c.values)
print(imp)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features=sc.fit_transform(features.values)
print(features)
pd.DataFrame(features)
