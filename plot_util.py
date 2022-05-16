
import matplotlib.pyplot as plt
from datetime import timedelta,datetime
import pandas as pd
import seaborn as sns
import numpy as np
def getSubPlot(row, column, h=5, w=5):
    fig, axes = plt.subplots(row, column, figsize=(int(column * w), int(row * h)))
    return fig, axes



def plotAroundDate(df,date,delta):
    df[pd.to_datetime(date)- delta:pd.to_datetime(date) + delta].plot(subplots=True)
    plt.show()
def plotAnomalyAroundDate(d, start, end, delta):
    ax= d[start- delta:end + delta].plot(c='g')
    d[pd.to_datetime(start):pd.to_datetime(end)].plot(c='r', ax=ax,subplots=True)
    plt.show()

def plotAroundDates(df,dates,delta):
  #  if type(dates) is not list:
    dates=dates.values.tolist()
    num=len(dates)
    rows=int(np.sqrt(num))
    columns=rows
    if rows*columns<num:
        columns+=1
    fig,axes=getSubPlot(rows, columns, 5, 5)
    for i in range(num):
        r=int(i/columns)
        c=i%columns
        df[pd.to_datetime(dates[i]) - delta:pd.to_datetime(dates[i])+ delta].plot(ax=axes[r][c],subplots=True)
    plt.show()

def plotAroundDates_twin(df,dates,delta):
  #  if type(dates) is not list:
    dates=dates.values.tolist()
    num=len(dates)
    rows=int(np.sqrt(num))
    columns=rows
    if rows*columns<num:
        columns+=1
    fig,axes=getSubPlot(rows, columns, 5, 5)
    for i in range(num):
        r=int(i/columns)
        c=i%columns
        df[df.columns[0]][pd.to_datetime(dates[i]) - delta:pd.to_datetime(dates[i])+ delta].plot(ax=axes[r][c])
        df[df.columns[1]][pd.to_datetime(dates[i]) - delta:pd.to_datetime(dates[i]) + delta].plot(ax=axes[r][c].twinx(),c='r')
    plt.show()

def plotAnomalyAroundDates(df,dates,delta):
    num=len(dates)
    rows=int(np.sqrt(num))
    columns=rows
    if rows*columns<num:
        columns+=1
    fig,axes=getSubPlot(rows, columns, 5, 5)
    for i in range(num):
        r=int(i/columns)
        c=i%columns
        df[dates[i][0] - delta:dates[i][1]+ delta].plot(c='k',ax=axes[r][c])
        df[dates[i][0]:dates[i][1]].plot(c='r',ax=axes[r][c])
    plt.show()

def plot1dDayes(df,startingDay=0,numOfDay=1):
    start=df.index[0].date()+timedelta(days=startingDay)
    end=start+timedelta(days=numOfDay)
    df[start:end].plot()
    for day in range(numOfDay):
        plt.axvline(start+timedelta(days=day), color='k', linestyle='--', alpha=0.2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.2)
    plt.show()

def plot1dDayHour(df,startingDay=0,startingHour=0,numOfHour=1):
    start=pd.to_datetime(df.index[0].date())+timedelta(days=startingDay)+timedelta(hours=startingHour)
    end=start+timedelta(hours=numOfHour)
    df[start:end].plot()
    for hour in range(numOfHour):
        plt.axvline(start+timedelta(hours=hour), color='k', linestyle='--', alpha=0.2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.2)
    plt.show()

def heatmap(df:pd.DataFrame):
    sns.heatmap(df)
    plt.show()