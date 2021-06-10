import pandas as pd
df = pd.read_csv("/content/AdultIncome.csv")
df.isnull().sum()
df  = pd.get_dummies(df, drop_first=True)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state= 0)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = dtc.score(x_test, y_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
cm2 = confusion_matrix(y_test, y_pred)
score2 = rfc.score(x_test, y_test)
