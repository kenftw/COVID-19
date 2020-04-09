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


filepath = 'C:/Users/hewit/Documents/GitHub/COVID-19/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv'
df = pd.read_csv(filepath)

italy = df.iloc[16,4:]

dates = pd.to_datetime(italy.index, format='%m/%d/%y')

ax = plt.gca()
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m'))
plt.plot(dates, italy, linewidth=0.5)
plt.show() 

