import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Создаём трейн и тест, строится модель России с учётом: Москва, Урал, Екатеринбург, Питер, близость к морю, близость к экватору.
target = lambda width, height: ((np.arctan(-width+60)+np.pi/2)/(np.pi+2) + 0.13*np.exp(-0.05*(170 - width)) + 0.7*np.exp(-0.05*(height-59)**2) * 0.7*np.exp(-0.05*(width-30)**2) + 0.7*np.exp(-0.05*(height-55)**2) * np.exp(-0.05*(width-37)**2) + 0.7*np.exp(-0.03*(height-56)**2) *np.exp(-0.03*(width-60)**2) + (width-105)**8/10**15.8 + ((height-53)*(np.sign(-height+53)+1)/2)**2/500-0.7)/1.5 + 0.5
width = np.linspace(20, 170, 500)
height = np.linspace(40, 75, 250)
w, h = np.meshgrid(width, height)

df = pd.DataFrame()
df["score"] = target(w.flatten(), h.flatten())
df["Lon"] = w.flatten()
df["Lat"] = h.flatten()
train, test = train_test_split(df, test_size=0.25)
test = test.drop("score", axis=1)

train.to_csv("train.csv")
test.to_csv("test.csv")

# Создаём доп. признаки: Расположение относительно Урала, близость к значимым городам (Москва, Питер, Екатеринбург), близость к левой, правой и нижней границе.
extra = pd.DataFrame()
extra_width = np.linspace(10, 175, 300)
extra_height = np.linspace(40, 75, 100)
extra_w, extra_h = np.meshgrid(extra_width, extra_height)
extra["Lon"] = extra_w.flatten()
extra["Lat"] = extra_h.flatten()
Ural = (extra.Lon < 60).apply(int)
extra["Before_Ural"] = Ural
Big_cities = np.exp(-(extra["Lat"]-59)**2) * np.exp(-(extra["Lon"]-30)**2) + np.exp(-(extra["Lat"]-55)**2) * np.exp(-(extra["Lon"]-37)**2) + np.exp(-(extra["Lat"]-56)**2) * np.exp(-(extra["Lon"]-60)**2)
extra["big_cities"] = Big_cities
dw = (np.max(extra_width)+ np.min(extra_width))/2
dh = (np.max(extra_height) + np.min(extra_height))/2
Close_to_border = (extra["Lon"]-dw)**8/10**16 + ((extra["Lat"]-dh)*(np.sign(-extra["Lat"]+dh)+1)/2)**2/500
extra["close_to_border"] = Close_to_border

extra.to_csv("features.csv")
