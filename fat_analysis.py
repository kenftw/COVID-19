# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 01:30:00 2020

@author: hewit
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('who_covid_19_situation_reports/who_covid_19_sit_rep_time_series/who_covid_19_sit_rep_time_series.csv')

confirmed_deaths_global = data.iloc[0,3:]

fig, ax = plt.subplots()

plt.plot(confirmed_deaths_global)
plt.title('Global Deaths')
plt.xlabel('Date')
plt.ylabel('Deaths')

every_nth = 30
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)



