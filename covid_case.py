# -*- coding: utf-8 -*-
from datetime import datetime
import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.linear_model import LinearRegression

import pmdarima as pm
from pmdarima.arima import ARIMA


filepath = 'C:/Users/hewit/Documents/GitHub/COVID-19/time_series_covid19_deaths_global.csv'
df = pd.read_csv(filepath)

italy = df.iloc[16,4:]

dates = pd.to_datetime(italy.index, format='%m/%d/%y')

ax = plt.gca()
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d/%m'))
#plt.plot(dates, italy)
plt.plot(italy)
plt.xticks
plt.show() 

#--------------- Predicting Canadian deaths ------------------
# isolate data pertainig to Canada
dfc = pd.DataFrame(df[(df['Country/Region']=='Canada')]).reset_index(drop=True)
ontario = dfc.iloc[7,4:]

ts = dfc.iloc[:,4:] # slice the numerical data into ts (forgo the provinces labels)

total = ts.sum(axis=0) # sum the columns to figure out total deaths in canada across provinces


# sarimax arima model fitting
stepwise_fit = pm.auto_arima(total, start_p=1, start_q=1,
                             max_p=4, max_q=4, max_d=2, m=12,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True,  # set to stepwise
                             maxiter=100)  



stepwise_fit.summary() # summarize model

pred = stepwise_fit.predict(n_periods=60, return_conf_int=True, alpha = 0.05) # make the prediction in to the future
p = pred[0]
ci = pred[1]


combined = np.concatenate((np.asarray(total), p))


# --------------- take log and re-do analysis --------------


