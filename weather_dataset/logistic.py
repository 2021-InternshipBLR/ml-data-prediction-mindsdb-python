import pandas as pd 
df = pd.read_csv("weatherAlbury.csv")
print("Size of weather DF is: ",df.shape)
df.head()

df = df.dropna()
print("New shape: ", df.shape)

df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes':1}, inplace = True)

x = df[["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday"]]
y = df.RainTomorrow

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Accuracy: ", score)


#Prediction from observations

observation = pd.read_csv("weatherObs.csv")
observation['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)
obsData = observation.values

y_pred = logreg.predict(obsData)
print(y_pred)
