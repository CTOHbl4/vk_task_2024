# Сначала запуск features.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from pickle import dump

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
features = pd.read_csv("features.csv", index_col=0)

# Присоединим features к train и test
rf = DecisionTreeClassifier(max_depth=1)
rf.fit(features[["Lon", "Lat"]], features["Before_Ural"])
train["Before_Ural"] = rf.predict(train[["Lon", "Lat"]])
test["Before_Ural"] = rf.predict(test[["Lon", "Lat"]])
rf = RandomForestRegressor(max_depth=11)
rf.fit(features[["Lon", "Lat"]], features["big_cities"])
train["big_cities"] = rf.predict(train[["Lon", "Lat"]])
test["big_cities"] = rf.predict(test[["Lon", "Lat"]])
rf = RandomForestRegressor(max_depth=11)
rf.fit(features[["Lon", "Lat"]], features["close_to_border"])
train["close_to_border"] = rf.predict(train[["Lon", "Lat"]])
test["close_to_border"] = rf.predict(test[["Lon", "Lat"]])

test.to_csv("open_data.csv")

# Обучим модель
y = train["score"]
X = train.drop("score", axis=1)
rf = RandomForestRegressor(max_depth=11)
rf.fit(X, y)
with open('model.pkl','wb') as f:
    dump(rf,f)
