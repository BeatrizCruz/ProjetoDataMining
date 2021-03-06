﻿# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:44:55 2019

@author: aSUS
"""
#IMPORTS
import pandas as pd
#!pip install modin[dask]
#import modin.pandas as pd # replaces pandas for parallel running, defaults to pandas when better method not available

#import sqlite3 # We ended up importing from csv and not from the sdatabase
import numpy as np
from matplotlib import pyplot as plt # For plots.
import math
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier # To treat missing values in children, education and firstPolicy variables.
import random # To treat missing values in livingArea variable.
from sklearn.neighbors import KNeighborsRegressor # To treat missing values in salary.
import statsmodels.api as sm # To create a linear regression for the variable salary.
from sklearn.linear_model import LinearRegression # to create a linear regression for the variable salary.
from sklearn import linear_model
from functools import reduce # K-classifier
# For the silhouette:
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm


#FUNCTIONS
def set_seed(my_seed):
    random.seed(my_seed)
    np.random.seed(my_seed)
my_seed=100
set_seed(my_seed)
#Diretorias:
file =f"./A2Z Insurance.csv"
#import csv file:
dfOriginal=pd.read_csv(file)
dfOriginal.head()

#Rename columns for easier access
colNames=list(dfOriginal.columns)
colNames
dfOriginal=dfOriginal.rename(columns={"Brithday Year": "birthday",
                                        "Customer Identity":"id",
                                        "First Policy´s Year":"firstPolicy",
                                        "Educational Degree":"education",
                                        "Gross Monthly Salary":"salary",
                                        "Geographic Living Area":"livingArea",
                                        "Has Children (Y=1)":"children",
                                        "Customer Monetary Value":"cmv",
                                        "Claims Rate":"claims",
                                        "Premiums in LOB: Motor":"lobMotor",
                                        "Premiums in LOB: Household":"lobHousehold",
                                        "Premiums in LOB: Health":"lobHealth",
                                        "Premiums in LOB:  Life":"lobLife",
                                        "Premiums in LOB: Work Compensations":"lobWork"})

#--------------------STUDY OF VARIABLES INDIVIDUALLY-----------------------
#Create columns for global flags with zeros
dfOriginal['Others'] = 0
dfOriginal['Errors'] = 0
dfOriginal['Outliers'] = 0
# 1. birthday:

# Plot birthday for a first visual analysis:
# plt.figure()
# dfOriginal['birthday'].value_counts().sort_index().plot()
# there might be strange values on the birthday that are distorting the plot.
# plt.show()
# plt.figure()
# dfOriginal['birthday'].hist() # The plot with the strange value is not perceptive
# plt.show()

#Let's check variable values:
print(dfOriginal['birthday'].value_counts().sort_index())# After checking variable values, we have a strange value on the birthday: 1028

# Create a new column to indicate strange values as 1 and normal values as 0 and a column that will reference all odd values for easier access
# There are few people alive that were born before 1900
dfOriginal['Strange_birthday'] = np.where(dfOriginal['birthday']<1900, 1,0)             #assign flag anomaly in birthday
dfOriginal['Others'] = np.where(dfOriginal['birthday']<1900, 1,dfOriginal['Others'])    #assign flag do not enter in model
dfOriginal['Errors'] = np.where(dfOriginal['birthday']<1900, 1,dfOriginal['Errors'])    #assign flag error
dfOriginal[['id','birthday']].loc[dfOriginal['Strange_birthday'] == 1] #what are the strange values
dfOriginal['Strange_birthday'].value_counts()   # Verify if the column was created as supposed

#Plot birthday variable with no strange values (where the new column equals zero):
plt.figure()
dfOriginal['birthday'][dfOriginal['Strange_birthday']==0].hist()
plt.show()


# Good Graph:
import plotly.offline as pyo
import plotly.graph_objs as go
df=pd.DataFrame(dfOriginal['birthday'][dfOriginal['Strange_birthday']==0].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Year'})
df=df.sort_values(by='Year')

data= go.Bar(x=df['Year'],y=df['birthday'])#,mode='markers')
layout = go.Layout(title='Birthday Variable',template='simple_white',
        xaxis=dict(title='Year',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

####################################################
# 2. firstPolicy
####################################################

# Plot firstPolicy for a first visual analysis:
# plt.figure()
# dfOriginal['firstPolicy'].value_counts().sort_index().plot() # there might be strange values on firstPolicy that are distorting the plot.
# plt.show()

#Check variable values:
dfOriginal['firstPolicy'].value_counts().sort_index() # there is a strange value: 53784

# Create a new column to indicate strange values as 1 and normal values as 0:
# Explain the choice of 2016: (...)
# Verify if the column was created as supposed
dfOriginal['strange_firstPolicy']=np.where(dfOriginal['firstPolicy']>2016, 1,0)
dfOriginal['Others']=np.where(dfOriginal['firstPolicy']>2016, 1,dfOriginal['Others'])       #assign flag do not enter in model
dfOriginal['Errors'] = np.where(dfOriginal['firstPolicy']>2016, 1,dfOriginal['Errors'])     #assign flag error

dfOriginal[['id','firstPolicy']].loc[dfOriginal['strange_firstPolicy'] == 1]                #what are the strange values

dfOriginal['strange_firstPolicy'].value_counts()


#Plot firstPolicy variable with no strange values (where the created column equals zero):

# Good Graph:
df=pd.DataFrame(dfOriginal['firstPolicy'][dfOriginal['strange_firstPolicy']==0].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Year'})
df=df.sort_values(by='Year')

data= go.Bar(x=df['Year'],y=df['firstPolicy'])#,mode='markers')
layout = go.Layout(title='First Policy Variable',template='simple_white',
        xaxis=dict(title='Year',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

################################################################
# 3. education (categorical variable)
#################################################################
# Create a variable to count the individuals per category:
counteducation=dfOriginal['education'].value_counts().sort_index()
counteducation

# Plot education variable:
# Good Graph:
df=pd.DataFrame(dfOriginal['education'].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Category'})
df=df.sort_values(by='Category')

data= go.Bar(x=df['Category'],y=df['education'])
layout = go.Layout(title='Education Variable',template='simple_white',
        xaxis=dict(title='Category',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)
# There is a considerable number of individuals with high education (BSc/MSc and PhD).
# The number of individuals having PhD is not that high. We will consider later if it makes sense to join the categories BSc/MSc and PhD in a unique category.

###############################################
# 4. SALARY
###############################################

# To study this variable as it has different values that are not easily repeated through individuals,
# instead of counting by value as done with the previous cases, we decided to make the cumulative to be used for plotting.
# Plot salary for a first visual analysis:
# plt.figure()
# dfOriginal['salary'].value_counts().sort_index().plot(style=".")
# plt.show()
# there might be strange values on salary that are distorting the plot.

# Check variable values and create a variable for that:
countSalary = dfOriginal['salary'].value_counts().sort_index() # there are 2 out of the box values: 34490, 55215
countSalary
# Create a new column to indicate outliers as 1 and normal values as 0:
# Explain chosen value for outliers (10000) (...)
# Verify if the column was created as supposed.
dfOriginal['Outliers_salary']=np.where(dfOriginal['salary']>10000, 1,0)                     #assign flag anomaly in salary
dfOriginal['Others'] = np.where(dfOriginal['salary']>10000, 1,dfOriginal['Others'])         #assign flag do not enter in model
dfOriginal['Outliers'] = np.where(dfOriginal['salary']>10000, 1,dfOriginal['Outliers'])     #assign flag outlier

dfOriginal[['id','salary']].loc[dfOriginal['Outliers_salary'] == 1]
dfOriginal['Outliers_salary'].value_counts()
#lower than minimum wage in 2016 (530€)
# Some values used during the report
dfOriginal[dfOriginal["salary"]<530].count().max()        #minimum wage 2016
dfOriginal[dfOriginal["salary"]<970].count().max()        #average wage 2016
dfOriginal[dfOriginal["salary"]<293].count().max()        #minimum wage 1998
dfOriginal[dfOriginal["salary"]<565].count().max()        #average wage 1998
# Create a variable with the cumulative salary values
#Plot the salary values and the cumulative values of salary
countSalaryCum = countSalary.cumsum()
countSalaryCum


# Plot salary non outliers values (where the created column equals zero):
# plt.figure()
# dfOriginal['salary'][dfOriginal['Outliers_salary']==0].hist()
# plt.show()
#plt.figure()    #as scatter
#dfOriginal['salary'][dfOriginal['Outliers_salary']==0].value_counts().sort_index().plot(style=".")
#plt.show()

# Good Graph as scatter:
df=pd.DataFrame(dfOriginal['salary'][dfOriginal['Outliers_salary']==0].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Salary'})
df=df.sort_values(by='Salary')

data= go.Scatter(x=df['Salary'],y=df['salary'],mode='markers')
layout = go.Layout(title='Salary Variable',template='simple_white',
        xaxis=dict(title='Salary Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

# As an histogram
df=pd.DataFrame(dfOriginal['salary'][dfOriginal['Outliers_salary']==0].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Salary'})
df=df.sort_values(by='Salary')

data= go.Histogram(x=df['Salary'],y=df['salary'])#,mode='markers')
layout = go.Layout(title='Salary Variable',template='simple_white',
        xaxis=dict(title='Salary Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)


# Good plot:
df=pd.DataFrame(dfOriginal['salary'][dfOriginal['Outliers_salary']==0].value_counts().sort_index().cumsum())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Salary','salary':'cumSalary'})

data= go.Scatter(x=df['Salary'],y=df['cumSalary'],mode='markers')
layout = go.Layout(title='Cumulative Salary Variable',template='simple_white',
        xaxis=dict(title='Cumulative Salary Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

# Check with log dist: as the usual behavior of a salary variable is a distribution with a heavy tail on the left side,
# usually it is applied a log transformation on the distribution in order to transform it to a normal distribution.

# Log distributon: not applicable as original distribution is already almost normal (it does not follow the usual behavior).
# Drop created column as it will not be used

################################################################
# 5. LIVING AREA (categorical variable)
################################################################
# Create a variable to count the individuals per category:
countlivingArea=dfOriginal['livingArea'].value_counts().sort_index()
countlivingArea

#Create a bar chart that shows the number of individuals per living area
#plt.figure()
#plt.bar(np.arange(len(countlivingArea.index)),countlivingArea)
#plt.xticks(np.arange(len(countlivingArea.index)),countlivingArea.index)
#plt.show()
# As we dont have any information on the location of each category of living area variable, we probably will not be able to suggest modifications such as joining categories.

# Good Graph:
df=pd.DataFrame(dfOriginal['livingArea'].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Category'})
df=df.sort_values(by='Category')

data= go.Bar(x=df['Category'],y=df['livingArea'])
layout = go.Layout(title='Living Area Variable',template='simple_white',
        xaxis=dict(title='Category',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

##############################################################
# 6. CHILDREN
##############################################################

# Create a variable to count the individuals per category:
countchildren=dfOriginal['children'].value_counts().sort_index()
countchildren

# Create a bar chart that shows the number of individuals with and without children
#plt.figure()
#plt.bar(np.arange(len(countchildren.index)),countchildren)
#plt.xticks(np.arange(len(countchildren.index)),countchildren.index)
#plt.show()
# There are more individuals with children that without.

# Good Graph:
df=pd.DataFrame(dfOriginal['children'].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Category'})
df=df.sort_values(by='Category')

data= go.Bar(x=df['Category'],y=df['children'])
layout = go.Layout(title='Children Variable',template='simple_white',
        xaxis=dict(title='Category',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

###################################################
# 7. cmv
###################################################

# To study this variable as it has different values that are not easily repeated through individuals,
# instead of counting by value, we decided to make the cumulative to plot, as done with the salary variable.

#Plot cmv for a first visual analysis:
plt.figure()
dfOriginal['cmv'].value_counts().sort_index().plot() # there might be strange values on cmv that are distorting the plot.
plt.show()
# The plot with the strange value is not perceptive

# Create a variable that counts individuals by cmv value to check cmv values
cmvValues=dfOriginal['cmv'].value_counts().sort_index()
# Create a boxplot to better visualize those values
# plt.figure()
# sb.boxplot(x=dfOriginal["cmv"])
# plt.show()

# Create a new column for negative outliers that indicates outliers as 1 and other values as 0. Clients that give huge losses to the company will have value 1 in this column.
# When creating the column put the 6 lower values that are represented on the boxplot (outliers) with value 1.
dfOriginal['df_OutliersLow_cmv'] = np.where(dfOriginal['cmv']<=cmvValues.index[5],1,0)
# Verify if the column was created as supposed
dfOriginal['df_OutliersLow_cmv'].value_counts()

# Create a box plot without the identified outliers.
#Check the ploted values in more detail.
# plt.figure()
# sb.boxplot(x = dfOriginal["cmv"][dfOriginal['df_OutliersLow_cmv'] == 0])
# plt.show()
cmvValues = dfOriginal['cmv'][dfOriginal['df_OutliersLow_cmv']==0].value_counts().sort_index()
cmvValues
# There are 6 lower values and 3 higher values that will be considered as outliers.

# Create a new column for positive outliers that indicates outliers as 1 and other values as 0. Clients that give huge profit to the company will have value 1 in this column.
# When creating this column put the 3 lower values that are represented on the boxplot (outliers) with value 1.
# Verify if the column was created as supposed
dfOriginal['df_OutliersHigh_cmv'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,0)
dfOriginal['df_OutliersHigh_cmv'].value_counts()

# Change the values of the new negative outliers to 1 in the df_OutliersLow_cmv column
# Verify if values were changed as supposed
dfOriginal['df_OutliersLow_cmv'] = np.where(dfOriginal['cmv']<=cmvValues.index[5],1,dfOriginal['df_OutliersLow_cmv'])

dfOriginal['df_OutliersLow_cmv'].value_counts()

# Create a box plot without the until now identified outliers:
#plt.figure()
#sb.boxplot(x = dfOriginal["cmv"][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)])
#plt.show()
#Check the ploted values in more detail:
cmvValues = dfOriginal['cmv'][(dfOriginal['df_OutliersLow_cmv']==0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)].value_counts().sort_index()
cmvValues
# There are 2 lower values that will be considered as outliers.

# Change the values of the new negative outliers to 1 in the df_OutliersLow_cmv column
dfOriginal['df_OutliersLow_cmv'] = np.where(dfOriginal['cmv']<=cmvValues.index[1],1,dfOriginal['df_OutliersLow_cmv'])
dfOriginal['Others'] = np.where(dfOriginal['cmv']<=cmvValues.index[1],1,dfOriginal['Others'])
# Verify if values were changed as supposed
dfOriginal['df_OutliersLow_cmv'].value_counts()


# Create a box plot without the until now identified outliers:
#plt.figure()
#sb.boxplot(x = dfOriginal["cmv"][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersLow_cmv'] == 0)])
#plt.show()
#Check the ploted values in more detail:
dfOriginal['Others'] = np.where(dfOriginal['cmv']<(-500), 1,dfOriginal['Others'])         #assign flag do not enter in model
dfOriginal['Others'] = np.where(dfOriginal['cmv']>2000, 1,dfOriginal['Others'])   #assign flag do not enter in model
dfOriginal['df_OutliersLow_cmv'] = np.where(dfOriginal['cmv']<(-500), 1,0)
dfOriginal['df_OutliersHigh_cmv'] = np.where(dfOriginal['cmv']>2000, 1,0)
dfOriginal['Outliers'] = np.where(dfOriginal['cmv']<(-500), 2,dfOriginal['Outliers'])       #assign flag outlier
dfOriginal['Outliers'] = np.where(dfOriginal['cmv']>2000, 3,dfOriginal['Outliers'])       #assign flag outlier


cmvValues = dfOriginal['cmv'][(dfOriginal['df_OutliersLow_cmv']==0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)].value_counts().sort_index()
cmvValues
#There are 3 higher values that will be considered as outliers.

# Change the values of the new positive outliers to 1 in the df_OutliersHigh_cmv column
#dfOriginal['df_OutliersHigh_cmv'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,dfOriginal['df_OutliersHigh_cmv'])

#Inquiries for knowing the values of the outliers
dfOriginal[['id','cmv']].loc[dfOriginal['df_OutliersLow_cmv'] == 1 ]
dfOriginal[['id','cmv']].loc[dfOriginal['df_OutliersHigh_cmv'] == 1]

#customers where cmv = -25  , possible aquisition cost? Let's find out
dfOriginal["cmv-25"]= np.where(dfOriginal['cmv']==(-25),1,0)
temp=dfOriginal[['id','cmv',"claims",'firstPolicy','birthday','lobMotor',"lobHousehold","lobHealth","lobLife","lobWork"]].loc[dfOriginal['cmv-25'] == 1] #Let's look at cmv -25
dfOriginal['Others'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,dfOriginal['Others'])
#dfOriginal['Others'] = np.where(dfOriginal['cmv']==(-25),1,dfOriginal['Others'])
temp2=dfOriginal[['firstPolicy','lobMotor',"lobHousehold","lobHealth","lobLife","lobWork",'claims','cmv']][dfOriginal['claims']==1]
temp3=dfOriginal[['firstPolicy','lobMotor',"lobHousehold","lobHealth","lobLife","lobWork",'claims']][dfOriginal['claims']==0]

# Verify if values were changed as supposed
dfOriginal['df_OutliersHigh_cmv'].value_counts()


##### Create a box plot without the until now identified outliers:
#plt.figure()
#sb.boxplot(x = dfOriginal["cmv"][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)])
#plt.show()

##### Plot with no outliers:
#plt.figure()
#dfOriginal['cmv'][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)].value_counts().sort_index().plot(style='.')
#plt.show()

##### Good Graph:
df=pd.DataFrame(dfOriginal['cmv'][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'cmv','cmv':'Individuals'})

data= go.Scatter(x=df['cmv'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='Customer Monetary Value Variable',template='simple_white',
        xaxis=dict(title='CMV Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

#####Same graph with no cmv=-25:
df=pd.DataFrame(dfOriginal['cmv'][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'cmv','cmv':'Individuals'})
df=df[df['cmv']!=-25]

data= go.Scatter(x=df['cmv'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='Customer Monetary Value Variable',template='simple_white',
        xaxis=dict(title='CMV Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)





############################################
# 8. claims
#############################################
#plt.figure()
#dfOriginal['claims'].value_counts().sort_index().plot() # there might be strange values on the claims that are distorting the plot.
#plt.show()
# plt.figure()
# dfOriginal['claims'].hist()
# plt.show()

# Check variable values:
valuesClaims = dfOriginal['claims'].value_counts().sort_index()
valuesClaims
# There is a small group of individuals who have high values of claims rate and that is the reason the plots are so distorted.

# Plot only values less than 4
df=dfOriginal.groupby(['claims'])['claims'].count()
df=pd.DataFrame(df, columns=['claims'])

#Let's remove outliers
# plt.figure()
# sb.boxplot(x = dfOriginal["claims"])
# plt.show()
# plt.figure()
# sb.violinplot(x = dfOriginal["claims"])
# plt.show()


# plt.figure()
# sb.boxplot(x = dfOriginal["claims"][dfOriginal['claims'] <4])
# plt.show()
# plt.figure()
# sb.violinplot(x = dfOriginal["claims"][dfOriginal['claims'] <4])
# plt.show()
#df=df[df.index<3]

dfOriginal['Outliers_claims']=np.where(dfOriginal['claims']>4, 1,0)                 #signal as outliers in claims
dfOriginal['Others']=np.where(dfOriginal['claims']>4, 1,dfOriginal['Others'])       #signal a global removable value
dfOriginal['Outliers'] = np.where(dfOriginal['claims']>4, 4,dfOriginal['Outliers']) #assign flag outlier
dfOriginal[['id','claims']].loc[dfOriginal['Outliers_claims'] == 1 ]
#plt.figure()
#df['claims'].sort_index().plot()
#plt.show()
# People who have a claims rate of 0 are the ones with which the company did not spend anything.
# People who have a claims rate between 0 and 1 (excluding) are the ones with which the company had profit. This means that the amount paid by the company was less than the premiums paid by the clients.
# People who have a claims rate of 1 are the ones with which the company had no profit nor losses.
# People who have a claims rate higher than 1 are the ones with which the company had losses.
#When premiums equals 0,claims=0
##### Good Graph:
df=pd.DataFrame(dfOriginal['claims'][(dfOriginal['Outliers_claims'] == 0)].value_counts().sort_index())
#there are still people with claims 1 when we take out CMV =-25
df=df.rename(columns={'claims':'Individuals'})
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'claims'})

data= go.Scatter(x=df['claims'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='Claims Variable',template='simple_white',
        xaxis=dict(title='Claims Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)


##### Without cmv=-25 Good Graph:
df=pd.DataFrame(dfOriginal['claims'][(dfOriginal['Outliers_claims'] == 0) & (dfOriginal['cmv'] != -25)].value_counts().sort_index())
#there are still people with claims 1 when we take out CMV =-25
df=df.rename(columns={'claims':'Individuals'})
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'claims'})

data= go.Scatter(x=df['claims'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='Claims Variable',template='simple_white',
        xaxis=dict(title='Claims Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

#Lets look at people that have a claims rate lower than 1:
df=dfOriginal.groupby(['claims'])['claims'].count()
df=pd.DataFrame(df, columns=['claims'])
df=df[df.index<1]
df=df.rename(columns={'claims':'Individuals'})
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'claims'})

df['Individuals'].sort_index().sum() #there are 8056 individuals that have a claims rate lower than 1

#let's look at people with claims =0
temp4=dfOriginal[['firstPolicy','lobMotor',"lobHousehold","lobHealth","lobLife","lobWork",'claims','cmv']][dfOriginal['claims']==0]

#Plot of the results
#plt.figure()
#df['claims'].sort_index().plot(style='.')
#plt.show()
# plt.figure()
# dfOriginal['claims'][dfOriginal['claims']<3].hist()
# plt.show()
# Good Graph:
data= go.Scatter(x=df['claims'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='Claims Variable',template='simple_white',
        xaxis=dict(title='Claims Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

#TODO: do this later after summing LOBS and correcting LOBS
# catClaims Variable
# Lets distinguish between individuals that give losses (losses), individuals that give profit (profits), individuals that do not give profits nor losses (neutrals) and individuals with which the company did not spend anything (investigate).
# The individuals that have a column value of 'investigate' need to be investigated later as we do not have any information about their premium values on this variable. We only know that the amount paid by the company is zero. We will need to study the premium value with the premium variables studied furtherahead.
dfOriginal['catClaims']=np.where(dfOriginal['claims']==0,'investigate','losses')
dfOriginal['catClaims']=np.where((dfOriginal['claims']>0)&(dfOriginal['claims']<1),'profits',dfOriginal['catClaims'])
dfOriginal['catClaims']=np.where((dfOriginal['claims']==1),'neutrals',dfOriginal['catClaims'])
#Check if the new column was created as wanted
dfOriginal['catClaims'].value_counts()
#############################################
# 9. lobMotor
#############################################
# Plot lobMotor for a first visual analysis:
# plt.figure()
# dfOriginal['lobMotor'].value_counts().sort_index().plot()
# plt.show()
# plt.figure()
# dfOriginal['lobMotor'].hist() # There might be few high values that are distorting the graphs
# plt.show()

# Check variable values:
valueslobMotor = dfOriginal['lobMotor'].value_counts().sort_index()
valueslobMotor
# plt.figure()
# sb.boxplot(x=dfOriginal["lobMotor"])
# plt.show()
# plt.figure()
# sb.violinplot(x=dfOriginal["lobMotor"])
# plt.show()

# Lets look for the fence high value of the box plot to define a from which value the lobMotor premium can be considered as an outlier.
q1=dfOriginal["lobMotor"].quantile(0.25)
q3=dfOriginal["lobMotor"].quantile(0.75)
iqr=q3-q1 #Interquartile range
fence_high=q3+1.5*iqr
fence_high
# Create a column that indicates if an individual is outlier or not (if it is, the column value will be 1)
dfOriginal['Outliers_lobMot']=np.where(dfOriginal['lobMotor']>750,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobMotor']>750,1,dfOriginal['Others'])            #assign flag do not enter in model
dfOriginal['Outliers'] = np.where(dfOriginal['lobMotor']>750, 5,dfOriginal['Outliers'])     #assign flag outlier
#TODO: Talk about cancelled
dfOriginal["TotalInsurance"]=np.where(dfOriginal["lobMotor"]>0,1,0)                         #CountInsurancesPaid
dfOriginal['CancelTotal'] = np.where(dfOriginal['lobMotor']<0, 1,0)                         #assign flag Cancelled
dfOriginal['CancelMotor'] = np.where(dfOriginal['lobMotor']<0, 1,0)                         #in this particular case, it's equal to cancelled
                                                                                            #but let's keep things consistent
dfOriginal['CancelTotal'].value_counts()
# Verify if column was created correctly:
dfOriginal['Outliers_lobMot'].value_counts()

# Create a box plot without the outliers:
# plt.figure()
# sb.boxplot(x = dfOriginal['lobMotor'][dfOriginal['Outliers_lobMot']==0])
# plt.show()


##### Good Scatter
df=pd.DataFrame(dfOriginal['lobMotor'][dfOriginal['Outliers_lobMot'] == 0].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'lobMotor','lobMotor':'Individuals'})

data= go.Scatter(x=df['lobMotor'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='lobMotor Variable',template='simple_white',
        xaxis=dict(title='lobMotor Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

##### With Histogram
df=pd.DataFrame(dfOriginal['lobMotor'][dfOriginal['Outliers_lobMot'] == 0].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'lobMotor','lobMotor':'Individuals'})

data= go.Histogram(x=df['lobMotor'],y=df['Individuals'])#,mode='markers')
layout = go.Layout(title='lobMotor Variable',template='simple_white',
        xaxis=dict(title='lobMotor Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

##########################################
# 10. lobHousehold
##########################################
# Plot lobHousehold for a first visual analysis:
#plt.figure()
#dfOriginal['lobHousehold'].value_counts().sort_index().plot()
#plt.show()
# plt.figure()
# dfOriginal['lobHousehold'].hist() # There might be few high values that are distorting the graphs
# plt.show()
valueslobHousehold = dfOriginal['lobHousehold'].value_counts().sort_index()
valueslobHousehold

# Box plot
# plt.figure()
# sb.boxplot(x=dfOriginal["lobHousehold"])
# plt.show()
# plt.figure()
# sb.violinplot(x=dfOriginal["lobHousehold"])
# plt.show()

#Lets define 3000 from which individuals are considered outliers
# Create a column that indicates individuals that are outliers and individuals that are not (the column values will be 1 if the individuals are considered outliers)
dfOriginal['Outliers_lobHousehold']=np.where(dfOriginal['lobHousehold']>2000,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobHousehold']>2000,1,dfOriginal['Others'])
dfOriginal['Outliers'] = np.where(dfOriginal['lobHousehold']>2000, 6,dfOriginal['Outliers'])
dfOriginal["TotalInsurance"]=np.where(dfOriginal["lobHousehold"]>0,dfOriginal["TotalInsurance"]+1,dfOriginal["TotalInsurance"])                         #CountInsurancesPaid
dfOriginal['CancelTotal'] = np.where(dfOriginal['lobHousehold']<0,dfOriginal['CancelTotal']+1,dfOriginal['CancelTotal'])                         #assign flag Cancelled
dfOriginal['CancelHouse'] = np.where(dfOriginal['lobHousehold']<0, 1,0)
dfOriginal['CancelHouse'].value_counts()
# Verify if column was created correctly:
dfOriginal['Outliers_lobHousehold'].value_counts()
#plt.figure()
#sb.boxplot(x = dfOriginal['lobHousehold'][dfOriginal['Outliers_lobHousehold']==0])
#plt.show()
# plt.figure()
# dfOriginal['lobHousehold'][dfOriginal['Outliers_lobHousehold']==0].value_counts().sort_index().plot()
# plt.show()
# We can observe that there are much more individuals with low values of household premiums than with high values, which makes sense because there are less houses that are expensive than cheaper ones.

# Good Graph:
df=pd.DataFrame(dfOriginal['lobHousehold'][dfOriginal['Outliers_lobHousehold'] == 0].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'lobHousehold','lobHousehold':'Individuals'})

data= go.Scatter(x=df['lobHousehold'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='LobHousehold Variable',template='simple_white',
        xaxis=dict(title='LobHousehold Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

data= go.Histogram(x=df['lobHousehold'],y=df['Individuals'])#,mode='markers')
layout = go.Layout(title='LobHousehold Variable',template='simple_white',
        xaxis=dict(title='LobHousehold Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

# Transformar em Logaritmo? (ver mais tarde)

###############################################
# 11. lobHealth
#############################################33
# plt.figure()
# dfOriginal['lobHealth'].value_counts().sort_index().plot()
# plt.show()
# plt.figure()
# dfOriginal['lobHealth'].hist() # There might be few high values that are distorting the graphs
# plt.show()

# Check variable values:
valueslobHealth  = dfOriginal['lobHealth'].value_counts().sort_index()
valueslobHealth

#Box plot
plt.figure()
sb.boxplot(x = dfOriginal['lobHealth'])
plt.show()

#Lets define 550 from which individuals are considered outliers
# Create a column that indicates individuals that are outliers and individuals that are not (the column values will be 1 if the individuals are considered outliers)
dfOriginal['Outliers_lobHealth']=np.where(dfOriginal['lobHealth']>550,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobHealth']>550,1,dfOriginal['Others'])
dfOriginal['Outliers'] = np.where(dfOriginal['lobHealth']>550, 7,dfOriginal['Outliers'])
dfOriginal["TotalInsurance"]=np.where(dfOriginal["lobHealth"]>0,dfOriginal["TotalInsurance"]+1,dfOriginal["TotalInsurance"])                         #CountInsurancesPaid
dfOriginal['CancelTotal'] = np.where(dfOriginal['lobHealth']<0,dfOriginal['CancelTotal']+1,dfOriginal['CancelTotal'])
dfOriginal['CancelHealth'] = np.where(dfOriginal['lobHealth']<0, 1,0)
dfOriginal['CancelHealth'].value_counts()
# Verify if column was created correctly:
dfOriginal['Outliers_lobHealth'].value_counts()

#Box plot without outliers
plt.figure()
sb.boxplot(x = dfOriginal['lobHealth'][dfOriginal['Outliers_lobHealth']==0])
plt.show()
# plt.figure()
# dfOriginal['lobHealth'][dfOriginal['Outliers_lobHealth']==0].value_counts().sort_index().plot()
# plt.show()

# We can observe that health premiums follows an approximatelly normal distribution, with long tails.
# On one hand, the health premiums are not as expensive as, for example, the house hold premiums, so more people have access to it.
# On the other hand, people consider health a primary preocupation and need.
# There are more people that invest a medium value on health premiums. Then, there are people who invest less (poorer people) and people who invest more (richer people)

##### Good Graphs
df=pd.DataFrame(dfOriginal['lobHealth'][dfOriginal['Outliers_lobHealth'] == 0].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'lobHealth','lobHealth':'Individuals'})

data= go.Scatter(x=df['lobHealth'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='lobHealth Variable',template='simple_white',
        xaxis=dict(title='lobHealth Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)
##### Good Histogram

data= go.Histogram(x=df['lobHealth'],y=df['Individuals'])#,mode='markers')
layout = go.Layout(title='lobHealth Variable',template='simple_white',
        xaxis=dict(title='lobHealth Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)
################################################
# 12. lobLife
################################################
# plt.figure()
# dfOriginal['lobLife'].value_counts().sort_index().plot()
# plt.show()
# plt.figure()
# dfOriginal['lobLife'].hist()
# plt.show()

# Check variable values:
valueslobLife  = dfOriginal['lobLife'].value_counts().sort_index()
valueslobLife

dfOriginal['Outliers_lobLife']=np.where(dfOriginal['lobLife']>375,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobLife']>375,1,dfOriginal['Others'])
dfOriginal['Outliers'] = np.where(dfOriginal['lobLife']>375, 9,dfOriginal['Outliers'])
dfOriginal["TotalInsurance"]=np.where(dfOriginal["lobLife"]>0,dfOriginal["TotalInsurance"]+1,dfOriginal["TotalInsurance"])                         #CountInsurancesPaid
dfOriginal['CancelTotal'] = np.where(dfOriginal['lobLife']<0,dfOriginal['CancelTotal']+1,dfOriginal['CancelTotal'])
dfOriginal['CancelLife'] = np.where(dfOriginal['lobLife']<0, 1,0)
dfOriginal['CancelLife'].value_counts()
# Box plot
# plt.figure()
# sb.boxplot(x = dfOriginal['lobLife'])
# plt.show()
# plt.figure()
# sb.violinplot(x = dfOriginal['lobLife'])
# plt.show()
# We decided not to consider outliers on this variable as there are no extreme individuals that influence the distribution (no individuals that highlight over the others).
# We can observe that more people invest less on life premiums.
# Transformar em Logaritmo? (ver mais tarde)

# Good Graph:
df=pd.DataFrame(dfOriginal['lobLife'].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'lobLife','lobLife':'Individuals'})

data= go.Scatter(x=df['lobLife'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='lobLife Variable',template='simple_white',
        xaxis=dict(title='lobLife Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

data= go.Histogram(x=df['lobLife'],y=df['Individuals'])#,mode='markers')
layout = go.Layout(title='lobLife Variable',template='simple_white',
        xaxis=dict(title='lobLife Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)


########################################
# 13. lobWork
#######################################
# plt.figure()
# dfOriginal['lobWork'].value_counts().sort_index().plot()
# plt.show()
# plt.figure()
# dfOriginal['lobWork'].hist()# There might be few high values that are distorting the graphs
# plt.show()

# Check variable values:
valueslobWork  = dfOriginal['lobWork'].value_counts().sort_index()
valueslobWork

#Box plot
plt.figure()
sb.boxplot(x = dfOriginal['lobWork'])
plt.show()
plt.figure()
sb.violinplot(x = dfOriginal['lobWork'],inner="quartile")
plt.show()
#Lets define 400 from which individuals are considered outliers
# Create a column that indicates individuals that are outliers and individuals that are not (the column values will be 1 if the individuals are considered outliers)
dfOriginal['Outliers_lobWork']=np.where(dfOriginal['lobWork']>400,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobWork']>400,1,dfOriginal['Others'])
dfOriginal['Outliers'] = np.where(dfOriginal['lobWork']>400, 8,dfOriginal['Outliers'])
dfOriginal["TotalInsurance"]=np.where(dfOriginal["lobWork"]>0,dfOriginal["TotalInsurance"]+1,dfOriginal["TotalInsurance"])                         #CountInsurancesPaid
dfOriginal['CancelTotal'] = np.where(dfOriginal['lobWork']<0,dfOriginal['CancelTotal']+1,dfOriginal['CancelTotal'])
dfOriginal['CancelWork'] = np.where(dfOriginal['lobWork']<0, 1,0)
dfOriginal['CancelWork'].value_counts()

# Verify if column was created correctly:
dfOriginal['Outliers_lobWork'].value_counts()

#Box plot without outliers
# plt.figure()
# sb.boxplot(x = dfOriginal['lobWork'][dfOriginal['Outliers_lobWork']==0])
# plt.show()
# plt.figure()
# dfOriginal['lobWork'][dfOriginal['Outliers_lobWork']==0].value_counts().sort_index().plot()
# plt.show()

# Good Graphs:
df=pd.DataFrame(dfOriginal['lobWork'][dfOriginal['Outliers_lobWork'] == 0].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'lobWork','lobWork':'Individuals'})

data= go.Scatter(x=df['lobWork'],y=df['Individuals'],mode='markers')
layout = go.Layout(title='lobWork Variable',template='simple_white',
        xaxis=dict(title='lobWork Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

data= go.Histogram(x=df['lobWork'],y=df['Individuals'])#,mode='markers')
layout = go.Layout(title='lobWork Variable',template='simple_white',
        xaxis=dict(title='lobWork Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

#################resume of confirmed canceled contracts in the last year############
dfOriginal['CancelTotal'].value_counts()

#######################################################################################################3
#Outliers
######################################################################################################

dfOriginal['Outliers'].value_counts()

dfOutliers=dfOriginal[["id","firstPolicy", "birthday", "education", "salary","livingArea","children", "cmv", "claims",
                       "lobMotor","lobHousehold", "lobHealth","lobLife","lobWork",
                       "Outliers_salary","df_OutliersLow_cmv","df_OutliersHigh_cmv","Outliers_claims","Outliers_lobMot",
                       "Outliers_lobHousehold","Outliers_lobHealth","Outliers_lobWork","Outliers_lobLife",
                        ]].loc[dfOriginal["Outliers"]>0]
dfOutliers.to_excel(f"./Outliers.xlsx", index=False, encoding='utf-8')
dfOutliers=dfOriginal.loc[dfOriginal["Others"]>0]

dfErrors=dfOriginal.loc[dfOriginal["Errors"]==1]

#-----------------CHECK INCOHERENCES------------------#

#Check if birthday is higher than First policy's year: 
agesList=[0,16,18]
countList=[]
for j in agesList:
    count_inc=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,'firstPolicy']-dfOriginal.loc[i,'birthday']<j: count_inc+=1
    countList.append(count_inc)
countList
#There are 1997 people that have a first policy before even being born. This does not make any sense.

# Create age column that calculates current ages of customers (useful to check the next incoherence).
# Check if there are people with less than 16 years old (including) who have children 
dfOriginal['age']= 2016-dfOriginal['birthday']
dfOutliers['age']= 2016-dfOutliers['birthday']
dfOriginal['age1998']= 1998-dfOriginal['birthday']
dfOutliers['age1998']= 1998-dfOutliers['birthday']
dfOriginal['children'][dfOriginal['age1998']<=16].value_counts()
# There are 31 people who are younger or equal to 16 years old and that have children and 16 younger or equal to 16 years old and that do not have children.
# At this age, in normal situations, there should be zero people with children. Even if there were some cases in these situations, 31 is a huge number.

# Check if people with age <9 years old have basic education.
# Check if people with age <16 years old have high school education. This would not make sense: in normal circumstances, people with less than 16 years old have not completed the high school education yet.
# Check if people with age <20 years old have Bsc/Msc education. This would not make sense: in normal circumstances, people with less than 20 years old have not completed the Bsc/ Msc education yet.
# Check if people with age <25 years old have Phd education. This would not make sense: in normal circumstances, people with less than 25 years old have not completed the Phd education yet.
dfOriginal['education'][dfOriginal['age1998']<9].value_counts()
dfOriginal[dfOriginal['age1998']<9] # There are no people with less than 9 years old.
dfOriginal['education'][dfOriginal['age1998']<12].value_counts() #There are 8 people who have a Phd with age less than 25 years old, which does not make sense.
dfOriginal['education'][dfOriginal['age1998']<16].value_counts() # People with less than 16 years old only have basic education (12 people), which makes sense.
dfOriginal['education'][dfOriginal['age1998']<20].value_counts() # People with less than 20 years old only have basic education (262 people) and high school education (81 people).
dfOriginal['education'][dfOriginal['age1998']<22].value_counts() #There are 8 people who have a Phd with age less than 25 years old, which does not make sense.
dfOriginal['education'][dfOriginal['age1998']<24].value_counts() #There are 8 people who have a Phd with age less than 25 years old, which does not make sense.

dfOriginal['education'][dfOriginal['age']<9].value_counts()
dfOriginal[dfOriginal['age']<9] # There are no people with less than 9 years old.
dfOriginal['education'][dfOriginal['age']<12].value_counts() #There are 8 people who have a Phd with age less than 25 years old, which does not make sense.
dfOriginal['education'][dfOriginal['age']<16].value_counts() # People with less than 16 years old only have basic education (12 people), which makes sense.
dfOriginal['education'][dfOriginal['age']<20].value_counts() # People with less than 20 years old only have basic education (262 people) and high school education (81 people).
dfOriginal['education'][dfOriginal['age']<22].value_counts() #There are 8 people who have a Phd with age less than 25 years old, which does not make sense.
dfOriginal['education'][dfOriginal['age']<24].value_counts()

# Check if people with less than 16 years old have a salary>0
dfOriginal['salary'][(dfOriginal['age']<16) & (dfOriginal['salary']>0)].count() #there are 12 people with less than 16 years old and that have a salary, which does not make sense. At these ages the expected salary value was expected to be zero, which means that we were expecting the output of this code line to be zero.
dfOriginal['salary'][(dfOriginal['age1998']<16) & (dfOriginal['salary']>0)].count()
# Check if people with less than 16 years old have a motor premium. This would not make sense as people with these ages do not have driving license.
dfOriginal['lobMotor'][dfOriginal['age']<16].count() # There are 12 people with less than 16 years old that have a motor premium, which does not make sense.
dfOriginal['lobMotor'][dfOriginal['age1998']<16].count()
# Check if people with less than 18 years old have a household premium (we defined the age 18 years old as the minimum age for a person to get a house).
dfOriginal['lobHousehold'][dfOriginal['age']<18].count() # There are 116 people younger than 18 years old who have a household premium, which does not make sense.
dfOriginal['lobHousehold'][dfOriginal['age1998']<18].count()
# Check if people with less than 16 years old have a work compensation premium, which does not make sense. The minimum age to start working is 16 years old.
dfOriginal['lobWork'][dfOriginal['age']<16].count() # There are 12 people younger than 16 years old that have a work compensation premium.
dfOriginal['lobWork'][dfOriginal['age1998']<16].count()
# Final Decision: drop birthday and age columns - they do not make any sense when considering other variables in the data set.

# Create a column for year salary (useful to check the next incoherence).
# Create a column with the total premiums (useful to check the next incoherence)
# Check if the 30% of the year salary is higher than the lobTotal
# Check if the 50% of the year salary is higher than the lobTotal
dfOriginal['yearSalary']=dfOriginal['salary']*12
dfOutliers['yearSalary']=dfOutliers['salary']*12
dfOriginal['lobTotal']=dfOriginal['lobMotor']+dfOriginal['lobHousehold']+dfOriginal['lobHealth']+dfOriginal['lobLife']+dfOriginal['lobWork']
dfOutliers['lobTotal']=dfOutliers['lobMotor']+dfOutliers['lobHousehold']+dfOutliers['lobHealth']+dfOutliers['lobLife']+dfOutliers['lobWork']
dfOriginal[dfOriginal['yearSalary']*0.3<dfOriginal['lobTotal']] # There are 14 people that spend more than 30% of the year salary in the total of premiums.
dfOriginal['id'][dfOriginal['yearSalary']*0.5<dfOriginal['lobTotal']].count() # There are 2 people that spend more than 50% of the year salary in the total of premiums, which migh be considered strange. It is not normal for a person spending more than 50% of the salary in premiums.

# Check if the year salary is higher than the lobTotal. If not, it does not make sense.
dfOriginal['id'][dfOriginal['yearSalary']<dfOriginal['lobTotal']].count() # There is one person that has a salary lower than the lobTotal, which does not make sense.
# We decided to add this customer to an incoherence column.
dfOriginal['incoherences']=np.where(dfOriginal['yearSalary']<dfOriginal['lobTotal'],1,0)
dfOutliers['incoherences']=np.where(dfOutliers['yearSalary']<dfOutliers['lobTotal'],1,0)
###############################################################################################3
#Errors
##############################################################################################3

dfOriginal['Errors'].value_counts()
dfErrors=dfOriginal.loc[dfOriginal["Errors"]==1]
dfOriginal["firstPolicy"].loc[dfOriginal['id']==9295]=None


#---------------CREATE A DATA FRAME TO WORK ON (WITH NO INCOHERENCES AND NO OUTLIERS)------------------------
# Take outliers, strange values and incoherences out.
# Drop columns that are not needed: because have all values zero and other reasons
dfWork=dfOriginal[dfOriginal['Others']==0]
dfWork=dfWork.drop(columns=['age', 'age1998','birthday','incoherences','Strange_birthday','Others','strange_firstPolicy','Outliers_salary','df_OutliersLow_cmv','df_OutliersHigh_cmv','Outliers_lobMot','Outliers_lobHousehold','Outliers_lobHealth','Outliers_lobWork'])

#----------------------------------MISSING VALUES----------------------------

# Create a column called 'Nan' to count the number of missing values by row
# Check the number of rows that have 0, 1, 2, 3 etc Nan values
dfNan=dfWork.drop(columns=['yearSalary','lobTotal'])
dfNan['Nan']=dfNan.isnull().sum(axis=1)
dfNan['Nan'].value_counts()
# Maximum number of Nan values is a row is 3.

# Count missing values by column:
dfNan.isnull().sum()

################################################################################
# LOB REPLACEMENT OF MISSING VALUES:

# We realized that a lot of missing values are related to the lob variables.
# We considered that these values would be equal to zero, as in an insurance company it is not normal not to register payments, unless they do not exist.
dfWork[dfWork['lobMotor'].isnull()]
dfWork['lobMotor'] = np.where(dfWork['lobMotor'].isnull(),0,dfWork['lobMotor'])
dfOutliers['lobMotor'] = np.where(dfOutliers['lobMotor'].isnull(),0,dfOutliers['lobMotor'])
dfWork[dfWork['lobHealth'].isnull()]
dfWork['lobHealth'] = np.where(dfWork['lobHealth'].isnull(),0,dfWork['lobHealth'])
dfOutliers['lobHealth'] = np.where(dfOutliers['lobHealth'].isnull(),0,dfOutliers['lobHealth'])
dfWork[dfWork['lobLife'].isnull()]
dfWork['lobLife'] = np.where(dfWork['lobLife'].isnull(),0,dfWork['lobLife'])
dfOutliers['lobLife'] = np.where(dfOutliers['lobLife'].isnull(),0,dfOutliers['lobLife'])
dfWork[dfWork['lobWork'].isnull()]
dfWork['lobWork'] = np.where(dfWork['lobWork'].isnull(),0,dfWork['lobWork'])
dfOutliers['lobWork'] = np.where(dfOutliers['lobWork'].isnull(),0,dfOutliers['lobWork'])

# Check again Nan values by row:
dfNan = dfWork.drop(columns = ['yearSalary', 'lobTotal'])
dfNan['Nan'] = dfNan.isnull().sum(axis=1)
dfNan['Nan'].value_counts()

# Recalculate the column lobTotal (as there are no Null values on the lob variables anymore)
dfWork['lobTotal']=dfWork['lobMotor']+dfWork['lobHousehold']+dfWork['lobHealth']+dfWork['lobLife']+dfWork['lobWork']
dfOutliers['lobTotal']=dfOutliers['lobMotor']+dfOutliers['lobHousehold']+dfOutliers['lobHealth']+dfOutliers['lobLife']+dfOutliers['lobWork']

# Check if lobTotal does not have Nan values
dfWork.isnull().sum()

#Check again Nan values per column:
dfNan.isnull().sum()

#########################################################################################
# LIVING AREA MISSING VALUES - Still have to decide what to do with this variable:

# In living area there is one null value. Lets try to check and treat it
# This is the row:
null=dfNan[dfNan['livingArea'].isnull()]
null

#Graph of lobhousehold and living area:
# plt.figure()
# sb.barplot(x="livingArea",y="lobHousehold",data=dfNan)
# plt.show()

# Check lobHousehold (>=0) by living area:
dfNan[dfNan['lobHousehold']>=0].groupby(by=["livingArea"])["lobHousehold"].mean().reset_index(name = 'Average_lobH')
# The average of lobHousehold does not differentiate a lot between different living areas.
dfNan[dfNan['lobHousehold']>=0].groupby(by=["livingArea"])["lobHousehold"].var().reset_index(name = 'Var_lobH')
# The variance does not differ a lot between living areas 1, 2 and 3, but the 4th one differs a lot from these 3 first areas.

# Living area might be determined by LobHousehold
# plt.figure()
# sb.boxplot(x='livingArea', y='lobHousehold', data=dfWork)
# plt.show()
# The boxplot shows that this does not happen

# Treat the Nan value through KNN
# Which variables should we use to predict the lobHousehold?
dfWork2=dfWork.dropna()

# plt.figure()
# sb.pairplot(dfWork2, vars=['firstPolicy','salary','cmv','claims','lobHousehold'], hue='livingArea')
# plt.show()
# plt.figure()
# sb.pairplot(dfWork2, vars=['lobMotor','lobHealth','lobLife','lobWork','lobTotal','yearSalary'], hue='livingArea')
# plt.show()
# There is no variable that explains the variable living area

dfWork2.groupby(by='livingArea').hist(alpha=0.4)
# We realized that the variable livingArea is not explained by any of the other variables. Besides, we do not have any information about this variable's categories. 

# As we probably will not use this variable on our analysis, we decided to input a random variable into the nan value on this variable.
# Check again null values by column
dfWork['livingArea']=np.where(dfWork['livingArea'].isna(),random.choice(dfWork['livingArea'][~dfWork['livingArea'].isna()]),dfWork['livingArea'])
dfWork.isna().sum()

############################################################################################
# Function to treat Nan Values through KNN:
def KNClassifier(dfWork,myDf,treatVariable,expVariables,K, weights,metric,p=2):
    """
    This function predicts a categorical variable through the KNN method (using KNeighborsClassifier). The arguments are the following:
    - dfWork: original data frame in which we want to introduce the final estimated values
    - myDf: data frame with an individuals' id column and all the variables that are going to be used (explained and explainable variables)
    - treatVariable: variable to predict (string type).
    - expVariables: list of variables that will be used to explain the treatVariable
    - K: number of neighbors to use.
    - weights: to choose the weight function to use (distance, uniform, callable)- for a more detailed explanation check the KNeighborsRegressor parameters.
    """
    varList=list(myDf)
    # df_inc: has Nan values on the treatVariable.
    # df_comp: no Nan values on the treatVariable.
    df_inc=myDf.loc[myDf[treatVariable].isna(),]
    df_comp=myDf[~myDf.index.isin(df_inc.index)]
    # change categorical variable to string to guarantee it can be a classifier.
    df_comp[treatVariable]=df_comp[treatVariable].astype('category')
    clf = KNeighborsClassifier(K,weights,metric=metric,p=p)
    # Use the df_comp data frame to train the model:
    trained_model = clf.fit(df_comp.loc[:,expVariables],
                        df_comp.loc[:,treatVariable])
    # Apply the trained model to the unknown data.
    # Drop treatVariable column from df_inc data frame.
    # Concat the df_inc data frame with the temp_df.
    # Introduce the data into the dfWork data frame.
    imputed_values = trained_model.predict(df_inc.drop(columns=[treatVariable,'id']))
    temp_df = pd.DataFrame(imputed_values.reshape(-1,1), columns = [treatVariable])
    df_inc = df_inc.drop(columns=[treatVariable])
    df_inc = df_inc.reset_index(drop=True)
    df_inc = pd.concat([df_inc, temp_df],
                              axis = 1,
                              ignore_index = True,
                              verify_integrity = False)
    df_inc.columns = varList
    df_inc = df_inc.drop(columns=expVariables)

    dfWork = reduce(lambda left,right: pd.merge(left, right, on='id', how='left'), [dfWork,df_inc])
    dfWork[treatVariable+'_x']=np.where(dfWork[treatVariable+'_x'].isna(),dfWork[treatVariable+'_y'],dfWork[treatVariable+'_x'])
    dfWork=dfWork.rename(columns={treatVariable+'_x':treatVariable})
    dfWork=dfWork.drop(columns=treatVariable+'_y')
    return dfWork
            
# Function to create a regression (and then if wanted to predict through it)
def Regression(myDf,indepVariables,depVariable,treatNa):
    """
    - myDf: data frame with an individuals' id column and all the variables that are going to be used (explained and explainable variables).
    - treatVariable: variable to predict (string type).
    - expVariables: list of variables that will be used to explain the treatVariable.
    - treatNa: boolean to define if it is to predict values with the created regression.
    """
    varList=list(myDf)  #get the list of variables
    # df_comp: dataframe without null values.
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    df_inc=myDf[pd.isnull(myDf).any(axis=1)]        #only rows with null values
    df_comp=myDf[~myDf.index.isin(df_inc.index)]    #only rows complete
    X = df_comp[indepVariables].values.reshape(-1,len(indepVariables))
    Y = df_comp[depVariable].values.reshape(-1,1)
    regressor=LinearRegression()
    regressor.fit(X,Y) #train
    y_pred=regressor.predict(X)# for metrics
    metricsDf= pd.DataFrame({"Actual":Y.flatten(),"Predicted":y_pred.flatten()}) # for comparison
#    df1 = metricsDf.head(50)                                                    #Let's plot a comparison for the first 50 values
#    df1.plot(kind='bar',figsize=(16,10))                                        #of predicted Vs actual
#    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
#    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#    plt.show()
    print('R^2:', metrics.r2_score(Y, y_pred))                                  #Let's print some metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, y_pred)))
    if (treatNa==True):
        df_inc[depVariable]=regressor.predict(df_inc[indepVariables])               #Actual prediction
        for myindex in df_inc["id"]:
            dfWork.loc[dfWork.id == myindex , depVariable] = df_inc.loc[df_inc.id==myindex,depVariable]
        return dfWork
 
############################################################################################
# CHILDREN - MISSING VALUES IMPUTATION

pd.pivot_table(dfWork,values='children', index=['education'],aggfunc=np.mean)
pd.pivot_table(dfWork,values='children', index=['livingArea'],aggfunc=np.mean)
# All education levels seem to have the same percentage of people with children. Therefore, we do not have to consider education to determine children.
# Same applied for livingArea

#Check null values on children:
dfWork['children'].isna().sum()     #There are 21 Nan values

# Which variables better explain the variable children?
# plt.figure()
# sb.pairplot(dfWork2, vars=['firstPolicy','salary','cmv','claims','lobHousehold'], hue='children')
# plt.show()
# plt.figure()
# sb.pairplot(dfWork2, vars=['lobMotor','lobHealth','lobLife','lobWork','lobTotal','yearSalary'], hue='children')
# plt.show()
#The variable salary explains the variable children.
#The variable lobMotor explains the variable children.
#Lets use the variables lobmotor and salary to explain the variable children and to treat the Nan values through the KNN:

# dfChildren: to treat Children Nan values
dfChildren=dfWork[['id','salary','lobMotor','lobHealth','children']]
dfChildren['children'][dfChildren['salary'].isna()].isna().sum()
dfChildren['children'][dfChildren['lobMotor'].isna()].isna().sum()
# There is no individual that has both salary and children null.
# There is no individual that has both LobMotor and children null (this would never happen as we have already treated null values for lob variables)

# Delete rows that have salary and/or lobMotor null.
dfChildren = dfChildren[~((dfChildren['salary'].isna())|(dfChildren['lobMotor'].isna()))]

# Apply the KNN Function:
dfWork=KNClassifier(dfWork=dfWork,myDf=dfChildren, treatVariable='children',
                    expVariables=['salary','lobMotor','lobHealth'],
                    K=5,weights='distance', metric='minkowski',p=1)
# We decided to use the manhattan distance because it considers the absolute distance in each variable.

#Check null values again:
dfWork.isna().sum()
###############################################################################################
# EDUCATION - MISSING VALUES IMPUTATION

#Check null values on education:
dfWork['education'].isna().sum()   #There are 17 Nan values

# Which variables better explain the variable education?
# sb.pairplot(dfWork2, vars=['firstPolicy','salary','cmv','claims','lobHousehold'], hue='education')
# plt.show()
# # sb.pairplot(dfWork2, vars=['lobMotor','lobHealth','lobLife','lobWork','lobTotal','yearSalary'], hue='education')
# # plt.show()
# The variables that better explain education (better discriminate the different classes of education) are: lobMotor, lobHousehold, salary
# Lets use the variables lobmotor, lobHousehold and salary to explain the variable education and to treat the Nan values through the KNN:

# dfEducation: to treat education Nan values
dfEducation=dfWork[['id','salary','lobMotor','lobHousehold','lobWork','lobLife','education']]

dfEducation['education'][dfEducation['salary'].isna()].isna().sum()
# There is one individual that has both salary and education null.

# Delete rows that have salary null.
dfEducation = dfEducation[~((dfEducation['salary'].isna()))]

# #Apply KNN function:
dfWork=KNClassifier(dfWork=dfWork,myDf=dfEducation, treatVariable='education', expVariables=['salary', 'lobMotor','lobHousehold','lobWork','lobLife'],K=5,weights='distance', metric='minkowski',p=1)
# Check null values again:
dfWork.isna().sum() # There is still 1 null value as expected because the individual has both salary and education null.

# Estimate the education value of this individual only with the variables lobMotor and lobHousehold.
dfEducation=dfWork[['id','lobMotor','lobHousehold','lobWork','lobLife','education']]

# Apply KNN function:
dfWork=KNClassifier(dfWork=dfWork, myDf=dfEducation, treatVariable='education', expVariables=['lobMotor','lobHousehold','lobWork','lobLife'], K=5,weights='distance', metric='minkowski',p=1)

# Check again nan values:
dfWork.isna().sum()

# Lets create a binary variable for education (this will be usefull to treat other posterior null values):
dfWork['binEducation']=dfWork['education']
dfOutliers['binEducation']=dfOutliers['education']
dfWork['binEducation']=np.where(((dfWork['binEducation']=='1 - Basic')|(dfWork['binEducation']=='2 - High School')),0,1)
dfOutliers['binEducation']=np.where(((dfOutliers['binEducation']=='1 - Basic')|(dfOutliers['binEducation']=='2 - High School')),0,1)
######################################################################################
# SALARY
# Which variables better explain salary?
# 1. Linear correlation: to check if there are linear correlations between variables.
# Through the heatmap, we can check that there is no variable that is highly linearly correlated with salary in absolute value. 
dfCorr=pd.DataFrame(dfWork,columns=['firstPolicy','salary','cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
dfCorrP=dfCorr.corr(method ='pearson')
fig, ax = plt.subplots(figsize=(10,10))
ax = sb.heatmap(dfCorrP, annot=True, fmt="0.3", square=True,  cbar_kws={'label': 'Colorbar'})
fig.suptitle('Heatmap - Linear Correlations (Pearson)')

# 2. Non linear correlations: to check if there are non linear correlations between variables:
# Through the heatmap, we can check that there is no variable that is highly non linearly correlated with salary in absolute value. 
dfCorrS=dfCorr.corr(method ='spearman')
fig, ax = plt.subplots(figsize=(10,10))
ax = sb.heatmap(dfCorrS, annot=True, fmt="0.3", square=True,  cbar_kws={'label': 'Colorbar'})
fig.suptitle('Heatmap - Non Linear Correlations (Spearman)')

# 3.  Plot the distribution of salary according to other variables: to check if there are variables that follow a specific distribution with the salary.
# To complement the point 2 (not actually necessary)
# fig=plt.figure()
# fig.suptitle('Distribution of Salary vs Other Variables')
# plt.subplot2grid((2,4),(0,0))
# plt.scatter(dfWork['cmv'], dfWork['salary'], alpha=0.5)
# plt.xlabel("CMV")
# plt.ylabel("Salary")
#
# plt.subplot2grid((2,4),(0,1))
# plt.scatter(dfWork['claims'], dfWork['salary'], alpha=0.5)
# plt.xlabel("Claims")
#
# plt.subplot2grid((2,4),(0,2))
# plt.scatter(dfWork['lobMotor'], dfWork['salary'], alpha=0.5)
# plt.xlabel("lobMotor")
#
# plt.subplot2grid((2,4),(0,3))
# plt.scatter(dfWork['lobHousehold'], dfWork['salary'], alpha=0.5)
# plt.xlabel("lobHousehold")
#
# plt.subplot2grid((2,4),(1,0))
# plt.scatter(dfWork['lobHealth'], dfWork['salary'], alpha=0.5)
# plt.xlabel("lobHealth")
# plt.ylabel("Salary")
#
# plt.subplot2grid((2,4),(1,1))
# plt.scatter(dfWork['lobLife'], dfWork['salary'], alpha=0.5)
# plt.xlabel("lobLife")
#
# plt.subplot2grid((2,4),(1,2))
# plt.scatter(dfWork['lobWork'], dfWork['salary'], alpha=0.5)
# plt.xlabel("lobWork")
#
# plt.subplot2grid((2,4),(1,3))
# plt.scatter(dfWork['lobTotal'], dfWork['salary'], alpha=0.5)
# plt.xlabel("lobTotal")
# plt.tight_layout(rect=[0.5, 0, 1, 1], h_pad=0.5)
# plt.plot()

# 4. Let's build a linear regression to check which variables are more significant to explain salary.
# The variables that coneptually would make more sense to explain salary are: lobHousehold, lobLife, lobWork, lobHousehold, lobHealth, firstPolicy, children and binEducation.
# Lets also include cmv and claims first and then take them out to see the difference that it makes on the R2 value.
dfSalary=dfWork[['id','salary','cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy','children','binEducation']]
Regression(myDf=dfSalary,indepVariables=['cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy','children','binEducation'],depVariable='salary',treatNa=False)
dfSalary=dfWork[['id','salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy','children','binEducation']]
Regression(myDf=dfSalary,indepVariables=['lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy','children','binEducation'],depVariable='salary',treatNa=False)
# Results: the R^2 are really low for both estimates done before (R^2=0.36)
# Therefore it does not make sense to decide the variables that better explain the salary through this poor regression estimated.
# If we had gotten a regression with a high R^2, we would even consider using it to estimate the null values. For this, we would use the estimated regression to estimate the null values, putting treatNa=True on the Regression function.
        # It would have been something like this:        
        # dfSalary=dfWork[['id','salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy','children']]
        # dfWork=Regression(myDf=dfSalary,indepVariables=['lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy','children'],depVariable='salary',treatNa=True)
# As we did not get a good R2 regression, we decided to treat the nan values through the KN Regression and to make a conceptually based decision on the variables that better explain salary.
# The variables that conceptually make sense explaining the variable salary are: LobHousehold, LobLife, lobWork, lobHousehold, lobHealth and firstPolicy, as said before (children and binEducation is binary so it cannot be used).
dfSalary=dfWork[['id','salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy']]
dfWork.isna().sum()
# Check if it happens the salary to be null at the same time as firstPolicy
dfSalary[(dfSalary['salary'].isna()) & (dfSalary['firstPolicy'].isna())]
# There are two observations in which salary and firstPolicy are null at the same time.
# Lets not consider these 2 observations for now and treat the other observations that have null salary:
# Lets also not consider observations with firstPolicy null as this is an explainable variable.
dfSalary=dfSalary[~(dfSalary['firstPolicy'].isna())]
dfSalary = dfSalary[~((dfSalary['salary'].isna()) & (dfSalary['firstPolicy'].isna()))]
dfSalary.isna().sum()

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance")
dfSalary2=dfSalary[['salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy']]
dfSalary2=pd.DataFrame(imputer.fit_transform(dfSalary2), columns=dfSalary2.columns)
# Check again nan values: 
dfSalary2.isna().sum()
# Replace column in the original data frame:
dfSalary2=dfSalary2[['salary']]
dfSalary2=dfSalary2.rename(columns={'salary':'salary_x'})
dfSalary=pd.DataFrame(pd.concat([dfSalary, dfSalary2],axis=1))
dfSalary=dfSalary[['id','salary_x']]
dfWork= reduce(lambda left,right: pd.merge(left, right, on='id', how='left'), [dfWork,dfSalary])
dfWork['salary']=np.where(dfWork['salary'].isna(),dfWork['salary_x'],dfWork['salary'])
dfWork=dfWork.drop(columns=['salary_x'])

#Check again Null values.
dfWork.isna().sum()
# As expected there are still two null values on salary. These are the one that we excluded because salary and firstPolicy were both null.

# Lets treat these two observations. For this we cannot use firstPolicy as an explainable variable. Lets just use the Lob variables:
dfSalary=dfWork[['id','salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork']]
dfSalary2=dfSalary[['salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork']]
dfSalary2=pd.DataFrame(imputer.fit_transform(dfSalary2), columns=dfSalary2.columns)
# Check again nan values: 
dfSalary2.isna().sum()
# Replace column in the original data frame:
dfSalary2=dfSalary2[['salary']]
dfSalary2=dfSalary2.rename(columns={'salary':'salary_x'})
dfSalary=pd.DataFrame(pd.concat([dfSalary, dfSalary2],axis=1))
dfSalary=dfSalary[['id','salary_x']]

# Replace column in the original data frame:
dfWork= reduce(lambda left,right: pd.merge(left, right, on='id', how='left'), [dfWork,dfSalary])
dfWork['salary']=np.where(dfWork['salary'].isna(),dfWork['salary_x'],dfWork['salary'])
dfWork=dfWork.drop(columns=['salary_x'])
#Check again Null values.
#Recalculate column yearSalary.
#Check again Null values.
dfWork.isna().sum() # zero null values on salary as expected
dfWork['yearSalary']=dfWork['salary']*12
dfOutliers['yearSalary']=dfOutliers['salary']*12
dfWork.isna().sum() # no null values on yearSalary

#dfWork=KNRegressor(dfWork=dfWork,myDf=dfSalary, treatVariable='salary',expVariables=['lobHousehold','lobMotor','lobHealth','lobLife','lobWork'],K=5,weights='uniform',metric="minkowski",p=1)  #1 for manhattan; 2 for euclidean

#############################################################################################################
# FIRST POLICY Impute Nulls

# Firstly, lets check which variables might better explain first policy.
# 1. Check linear correlations through the Pearson correlations. - already previously created heatmap.
# There is no variable linearly correlated with firstPolicy.
# 2. Check non linear correlations through the Spearman correlations. - already previously created heatmap.
# There is no variable non linearly correlated with firstPolicy.
# 3. Lets check non linear correlations visually (not necessary, just a complement to point 2.)
            
#fig=plt.figure()
#fig.suptitle('Distribution of firstPolicy vs Other Variables')
#plt.subplot2grid((2,4),(0,0))
#plt.scatter(dfWork['cmv'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("CMV")
#plt.ylabel("fisrtPolicy")
#
#plt.subplot2grid((2,4),(0,1))
#plt.scatter(dfWork['claims'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("Claims")
#
#plt.subplot2grid((2,4),(0,2))
#plt.scatter(dfWork['lobMotor'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("lobMotor")
#
#plt.subplot2grid((2,4),(0,3))
#plt.scatter(dfWork['lobHousehold'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("lobHousehold")
#
#plt.subplot2grid((2,4),(1,0))
#plt.scatter(dfWork['lobHealth'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("lobHealth")
#plt.ylabel("fisrtPolicy")
#
#plt.subplot2grid((2,4),(1,1))
#plt.scatter(dfWork['lobLife'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("lobLife")
#
#plt.subplot2grid((2,4),(1,2))
#plt.scatter(dfWork['lobWork'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("lobWork")
#
#plt.subplot2grid((2,4),(1,3))
#plt.scatter(dfWork['salary'], dfWork['firstPolicy'], alpha=0.5)
#plt.xlabel("salary")
#plt.tight_layout(rect=[0.5, 0, 1, 1], h_pad=0.5)
#plt.plot()
# Visually we can also observe what was stated before, that there is no variable non linearly correlated with first policy.

# 4. Let's build a linear regression to check which variables are more significant to explain firstPolicy.
# The variables that conceptually would make more sense to explain firstPolicy are: lobHousehold, lobLife, lobWork, lobHousehold, lobHealth, salary, children and binEducation. Lets use them to build a regression:
# Lets also include cmv and claims first and then take them out to see the difference that it makes on the R2 value.
dfFirstPolicy=dfWork[['id','firstPolicy','salary','cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','children','binEducation']]
Regression(myDf=dfFirstPolicy,indepVariables=['salary','cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','children','binEducation'],depVariable='firstPolicy',treatNa=False)
dfFirstPolicy=dfWork[['id','firstPolicy','salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','children','binEducation']]
Regression(myDf=dfFirstPolicy,indepVariables=['salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','children','binEducation'],depVariable='firstPolicy',treatNa=False)
# Results: the R^2 are really low for both estimates done before (R^2=0.001 and R^2=0.002)
# Therefore, it does not make sense to decide the variables that better explain the salary or to estimate the fisrtPolicy values through this poor regression estimated.

# Lets apply the KNRegressor technique: 
# Conceptually we have said that the variables that might explain firstPolicy are: lobHousehold, lobLife, lobWork, lobHousehold, lobHealth and salary (children and binEducation excluded as they are binary)
dfFirstPolicy=dfWork[['id','firstPolicy','salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork']]
dfFirstPolicy2=dfFirstPolicy[['salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','firstPolicy']]
dfFirstPolicy2=pd.DataFrame(imputer.fit_transform(dfFirstPolicy2), columns=dfFirstPolicy2.columns)
# Check again nan values: 
dfFirstPolicy2.isna().sum()
# Replace column in the original data frame:
dfFirstPolicy2=dfFirstPolicy2[['firstPolicy']]
dfFirstPolicy2=dfFirstPolicy2.rename(columns={'firstPolicy':'firstPolicy_x'})
dfFirstPolicy=pd.DataFrame(pd.concat([dfFirstPolicy, dfFirstPolicy2],axis=1))
dfFirstPolicy=dfFirstPolicy[['id','firstPolicy_x']]

# Replace column in the original data frame:
dfWork = reduce(lambda left,right: pd.merge(left, right, on='id', how='left'), [dfWork,dfFirstPolicy])
dfWork['firstPolicy']=np.where(dfWork['firstPolicy'].isna(),dfWork['firstPolicy_x'],dfWork['firstPolicy'])
dfWork = dfWork.drop(columns=['firstPolicy_x'])

#Verify null values
dfWork.isna().sum()

# round the firstPolicy values to zero digits 
dfWork['firstPolicy']=round(dfWork['firstPolicy'], 0)

###############################################################################################
#-------------------------------------------NEW VARIABLES----------------------------------------------------

# lobTotal (already created)
# yearSalary (already created)
# binEducation (already created)
# catClaims Variable (already created)
# Ratios lobs 
dfWork['motorRatioLOB']=np.where(dfWork['lobTotal']==0,0,dfWork["lobMotor"]/dfWork['lobTotal'])
dfOutliers['motorRatioLOB']=np.where(dfOutliers['lobTotal']==0,0,dfOutliers["lobMotor"]/dfOutliers['lobTotal'])
dfWork['householdRatioLOB']=np.where(dfWork['lobTotal']==0,0,dfWork["lobHousehold"]/dfWork['lobTotal'])
dfOutliers['householdRatioLOB']=np.where(dfOutliers['lobTotal']==0,0,dfOutliers["lobHousehold"]/dfOutliers['lobTotal'])
dfWork['healthRatioLOB']=np.where(dfWork['lobTotal']==0,0,dfWork["lobHealth"]/dfWork['lobTotal'])
dfOutliers['healthRatioLOB']=np.where(dfOutliers['lobTotal']==0,0,dfOutliers["lobHealth"]/dfOutliers['lobTotal'])
dfWork['lifeRatioLOB']=np.where(dfWork['lobTotal']==0,0,dfWork["lobLife"]/dfWork['lobTotal'])
dfOutliers['lifeRatioLOB']=np.where(dfOutliers['lobTotal']==0,0,dfOutliers["lobLife"]/dfOutliers['lobTotal'])
dfWork['workCRatioLOB']=np.where(dfWork['lobTotal']==0,0,dfWork["lobWork"]/dfWork['lobTotal'])
dfOutliers['workCRatioLOB']=np.where(dfOutliers['lobTotal']==0,0,dfOutliers["lobWork"]/dfOutliers['lobTotal'])

# lobTotal/salary
dfWork['ratioSalaryLOB']=dfWork['lobTotal']/dfWork['salary']
dfOutliers['ratioSalaryLOB']=dfOutliers['lobTotal']/dfOutliers['salary']
# Years has been a customer= 1998-firstPolicy compare with 2016
dfWork['YearsWus1998']=1998-dfWork['firstPolicy']
dfOutliers['YearsWus1998']=1998-dfOutliers['firstPolicy']
dfWork['YearsWus2016']=2016-dfWork['firstPolicy']
dfOutliers['YearsWus2016']=2016-dfOutliers['firstPolicy']

dfWork['CMV_Mean_corrected']=(dfWork['cmv']+25)
dfOutliers['CMV_Mean_corrected']=(dfOutliers['cmv']+25)
##### create categorical values of cmv corrected
dfWork['catCMV_Mean_corrected']=np.where(dfWork['cmv']<-25,'losses',None)
dfOutliers['catCMV_Mean_corrected']=np.where(dfOutliers['cmv']<-25,'losses',None)
dfWork['catCMV_Mean_corrected']=np.where(dfWork['cmv']>-25,'profitable',dfWork['catCMV_Mean_corrected'])
dfOutliers['catCMV_Mean_corrected']=np.where(dfOutliers['cmv']>-25,'profitable',dfOutliers['catCMV_Mean_corrected'])
dfWork['catCMV_Mean_corrected']=np.where(dfWork['cmv']==-25,'neutrals',dfWork['catCMV_Mean_corrected'])
dfOutliers['catCMV_Mean_corrected']=np.where(dfOutliers['cmv']==-25,'neutrals',dfOutliers['catCMV_Mean_corrected'])
#check cancel o see if categorical cancel is a thing
df=pd.DataFrame(dfWork[["CancelTotal","lobMotor","lobHousehold","lobHealth","lobLife","lobWork"]][dfWork['CancelTotal'] > 0 ])

#TODO: Plot new variables: cancel, YearsWus1998, YearsWus2016,

df=pd.DataFrame(dfWork['YearsWus1998'].value_counts())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'Years'})
df=df.sort_values(by='Years')

data= go.Bar(x=df['Years'],y=df['YearsWus1998'])#,mode='markers')
layout = go.Layout(title='Years Since First Policy',template='simple_white',
        xaxis=dict(title='Years',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)


##### Plot CMV with years with us
df=dfWork[['YearsWus1998','cmv']][(dfOriginal['strange_firstPolicy']==0) &
                                   (dfOriginal['df_OutliersLow_cmv'] == 0 )&
                                   (dfOriginal['df_OutliersHigh_cmv'] == 0)]#.value_counts())
df.reset_index(level=0, inplace=True)
df=df.sort_values(by='YearsWus1998')

data= go.Scatter(x=df['YearsWus1998'],y=df['cmv'],mode='markers')
layout = go.Layout(title='Years Since First Policy',template='simple_white',
        xaxis=dict(title='YearsWus',showgrid=True),yaxis=dict(title='cmv',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

#plot catCMVcorrected
df=pd.DataFrame(dfWork['catCMV_Mean_corrected'].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'catCMV_Mean_corrected','catCMV_Mean_corrected':'Individuals'})


data= go.Bar(x=df['catCMV_Mean_corrected'],y=df['Individuals'])
layout = go.Layout(title='Customer Monetary Value Mean Corrected Variable',template='simple_white',
        xaxis=dict(title='CMV Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

####################################################################################################################
#---------------------------------------------- MULTIDIMENSIONAL OUTLIERS -------------------------------------------------#
dfMultiOut=dfWork[['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','lobTotal','motorRatioLOB','householdRatioLOB','healthRatioLOB','lifeRatioLOB','workCRatioLOB','YearsWus1998','CMV_Mean_corrected','ratioSalaryLOB']]
# Min max: outliers already treated and for variables to be at the same scale.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
multiNorm = scaler.fit_transform(dfMultiOut)
multiNorm = pd.DataFrame(multiNorm, columns = dfMultiOut.columns)
multiNorm.isna().sum()
my_seed=100
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram,linkage

# Apply k-means with the k as the square root of the obs number.
K=int(round(math.sqrt(10261),0))
kmeans = KMeans(n_clusters=K, 
                random_state=0,
                n_init = 20, 
                max_iter = 300,
                init='k-means++').fit(multiNorm)

# Check the Clusters (Centroids).
multiClusters=kmeans.cluster_centers_
multiClusters

labelsKmeans = pd.DataFrame(kmeans.labels_)
labelsKmeans.columns = ['LabelsKmeans']
labelsKmeans

outClientsCluster = pd.DataFrame(pd.concat([multiNorm, labelsKmeans],axis=1), 
                        columns=['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','LabelsKmeans'])

# Hierarchical clustering:
# Apply Hierarchical clustering with 30 clusters over the 100 clusters created by the k - means
from sklearn.cluster import AgglomerativeClustering
hClustering = AgglomerativeClustering(n_clusters = 30,
                                      affinity = 'euclidean',
                                      linkage = 'ward')

# With 30 clusters the normal behavior would be to have 300 observations in each cluster.
multiHC = hClustering.fit(multiClusters)

labelsHC = pd.DataFrame(multiHC.labels_)
labelsHC.columns =  ['LabelsHC']
labelsHC.reset_index(level=0, inplace=True)
labelsHC=labelsHC.rename(columns={'index':'LabelsKmeans'})

outClientsCluster=outClientsCluster.merge(labelsHC, left_on='LabelsKmeans', right_on='LabelsKmeans')
outClientsCluster['LabelsHC'].value_counts().sort_values()

# Dendogram to check when the observations of the smallest clusters get together. If they join in a cluster near the end there is a higher evidence that these might be outliers.
Z = linkage(multiClusters,
                method = 'ward')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (over k-means)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z,
           truncate_mode='none',
           p=40,
           orientation = 'top',
           leaf_rotation=45,
           leaf_font_size=10,
           show_contracted=True,
           show_leaf_counts=True,color_threshold=50, above_threshold_color='k',no_labels =True)

######### On k=10 there is a late individual that joins a cluster. Lets check kmeans+ hc with nHC=10.
dfMultiOut=dfWork[['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','lobTotal','motorRatioLOB','householdRatioLOB','healthRatioLOB','lifeRatioLOB','workCRatioLOB','YearsWus1998','CMV_Mean_corrected','ratioSalaryLOB']]
# Min max: outliers already treated and for variables to be at the same scale.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
multiNorm = scaler.fit_transform(dfMultiOut)
multiNorm = pd.DataFrame(multiNorm, columns = dfMultiOut.columns)
multiNorm.isna().sum()
my_seed=100
from sklearn.cluster import KMeans

# Apply k-means with the k as the square root of the obs number.
K=int(round(math.sqrt(10261),0))
kmeans = KMeans(n_clusters=K, 
                random_state=0,
                n_init = 20, 
                max_iter = 300,
                init='k-means++').fit(multiNorm)

# Check the Clusters (Centroids).
multiClusters=kmeans.cluster_centers_
multiClusters

labelsKmeans = pd.DataFrame(kmeans.labels_)
labelsKmeans.columns =  ['labelsKmeans']
labelsKmeans

outClientsCluster = pd.DataFrame(pd.concat([multiNorm, labelsKmeans],axis=1), 
                        columns=['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','labelsKmeans'])

# Hierarchical clustering:
# Apply Hierarchical clustering with 30 clusters over the 100 clusters created by the k - means
from sklearn.cluster import AgglomerativeClustering
hClustering = AgglomerativeClustering(n_clusters = 10,
                                      affinity = 'euclidean',
                                      linkage = 'ward')

# With 30 clusters the normal behavior would be to have 300 observations in each cluster.
multiHC = hClustering.fit(multiClusters)

labelsHC = pd.DataFrame(multiHC.labels_)
labelsHC.columns =  ['labelsHC']
labelsHC.reset_index(level=0, inplace=True)
labelsHC=labelsHC.rename(columns={'index':'labelsKmeans'})

outClientsCluster=outClientsCluster.merge(labelsHC, left_on='labelsKmeans', right_on='labelsKmeans')
outClientsCluster['labelsHC'].value_counts().sort_values()


plt.figure()
sb.pairplot(dfWork, vars=['lobMotor','lobHousehold','lobHealth','lobLife','lobWork','lobTotal'])
plt.show()


# Pair plot to check ni dimensional outliers:
plt.figure()
sb.pairplot(dfWork, vars=['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','lobTotal','motorRatioLOB','householdRatioLOB','healthRatioLOB','lifeRatioLOB','workCRatioLOB','YearsWus1998','CMV_Mean_corrected','ratioSalaryLOB'])
plt.show()

###########################################################################################################
#---------------------------------------------- CLUSTERS -------------------------------------------------#
###########################################################################################################
#---------------------------------------------- FUNCTIONS ------------------------------------------------#

#####################################    
# K-MEANS
#####################################
def kmeans_funct(dfKmeans, dfNorm, n, returndf=False):
    """df is a data frame with all the variables we want to use to apply the k-means"""
    # Apply k-means
    kmeans = KMeans(n_clusters=n,
                    random_state=0,
                    n_init = 20,
                    max_iter = 300,
                    init='k-means++').fit(dfNorm)
    labelsKmeans = pd.DataFrame(kmeans.labels_)
    labelsKmeans.columns =  ['labelsKmeans']
    dfKmeans = pd.DataFrame(pd.concat([dfKmeans, labelsKmeans],axis=1))
    print('Inertia: ' + str(kmeans.inertia_))
    
    # Box Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (K-means)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.boxplot(x='labelsKmeans', y=i, data=dfKmeans)
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Violin Plots:
    lista=list(dfNorm.columns)
    fig2=plt.figure()
    fig2.suptitle('Violin Plots by Variable and Cluster (K-means')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.violinplot(x='labelsKmeans', y=i, data=dfKmeans, scale='width')
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    #Silhouette Graph:
    silhouette_avg = silhouette_score(dfNorm, kmeans.labels_) 
    print("For number of cluster (k) =", n,
              "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dfNorm, kmeans.labels_)
    cluster_labels = kmeans.labels_
    y_lower = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, dfNorm.shape[0] + (n + 1) * 10]) 
    
    for i in range(n):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()    
        size_cluster_i=ith_cluster_silhouette_values. shape[0]
        y_upper = y_lower + size_cluster_i    
        color = cm.nipy_spectral(float(i) / n)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color,
                              edgecolor=color, 
                              alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        ax.set_title("Silhouette Plot")
        ax.set_xlabel("The Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()
    if (returndf==True):
        return dfKmeans 
#####################################    
# K-means+ Hierarchical:
#####################################
def kmeansHC_funct(dfNorm,dfKmeansHC,nkmeans,nHC,returndf=False):
    kmeans = KMeans(n_clusters=nkmeans, 
                    random_state=0,
                    n_init = 20, 
                    max_iter = 300,
                    init='k-means++').fit(dfNorm)
    
    # Centroids:
    kmeansHCCentroidsEngage=kmeans.cluster_centers_
    labelsKmeansHC = pd.DataFrame(kmeans.labels_)
    labelsKmeansHC.columns =  ['labelsKmeansHC']
    
    dfKmeansHC = pd.DataFrame(pd.concat([dfKmeansHC, labelsKmeansHC],axis=1))
    
    # Create a dendogram to check how many of the nkmeans clusters should be retained:
    Z = linkage(kmeansHCCentroidsEngage,
                method = 'ward')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram (over k-means)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z,
               truncate_mode='none',
               p=40,
               orientation = 'top',
               leaf_rotation=45,
               leaf_font_size=10,
               show_contracted=True,
               show_leaf_counts=True,color_threshold=50, above_threshold_color='k',no_labels =True) # Evidence to retain 2, 3 or 4 clusters
    
    # Apply Hierarchical Clustering to the formed centroids: 
    hClustering = AgglomerativeClustering(n_clusters = nHC, 
                                          affinity = 'euclidean',
                                          linkage = 'ward').fit(kmeansHCCentroidsEngage)
    
    labelsKmeansHC2 = pd.DataFrame(hClustering.labels_)
    labelsKmeansHC2.columns =  ['labelsKmeansHC2']
    labelsKmeansHC2.reset_index(level=0, inplace=True)
    labelsKmeansHC2=labelsKmeansHC2.rename(columns={'index':'labelsKmeansHC'})
    
    # Join the new clusters to the data frame dfKmeansHC with merge through the 'LabelsKmeansHC'
    dfKmeansHC=dfKmeansHC.merge(labelsKmeansHC2, left_on='labelsKmeansHC', right_on='labelsKmeansHC')
    
    # Box Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (K-means + Hierarchical)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.boxplot(x='labelsKmeansHC2', y=i, data=dfKmeansHC)
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Violin Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (K-means + Hierarchical)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.violinplot(x='labelsKmeansHC2', y=i, data=dfKmeansHC, scale='width')
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Silhouette Graph:   
    silhouette_avg = silhouette_score(kmeansHCCentroidsEngage,hClustering.labels_)
    print("For number of cluster (k) =", nHC,
              "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(kmeansHCCentroidsEngage,hClustering.labels_)
    cluster_labels = hClustering.labels_
    y_lower = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, dfNorm.shape[0] + (nHC + 1) * 10]) 
    
    for i in range(nHC):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()    
        size_cluster_i=ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i    
        color = cm.nipy_spectral(float(i) / nHC)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color,
                              edgecolor=color, 
                              alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        ax.set_title("Silhouette Plot")
        ax.set_xlabel("The Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()
    
    if (returndf==True):
        return dfKmeansHC 

#####################################    
#SOM + Hierarchical:
#####################################
def SomHC_funct(dfNorm,dfSomHC,names,nHC,returndf):
    from sompy.sompy import SOMFactory
    sm = SOMFactory().build(data = dfNorm,
                   mapsize=(10,10),
                   normalization = 'var',
                   initialization='random',#'random', 'pca'
                   component_names=names,
                   lattice='hexa',#'rect','hexa'
                   training ='seq' )
    
    sm.train(n_job=4, #to be faster
             verbose='info', # to show lines when running
             train_rough_len=30, # first 30 steps are big (big approaches) - move 50%
             train_finetune_len=100) # small steps - move 1%
    
    labelsSomHC = pd.DataFrame(sm._bmu[0]) 
    labelsSomHC.columns = ['labelsSomHC']
    
    dfSomHC = pd.DataFrame(pd.concat([dfSomHC, labelsSomHC],axis=1))    
    # Get centroids:
    somHCCentroidsEngage=sm.codebook.matrix
    
    # Create a dendogram to check how many of the formed clusters should be retained:
    Z = linkage(somHCCentroidsEngage,
                method = 'ward')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z,
               truncate_mode='none',
               p=40,
               orientation = 'top',
               leaf_rotation=45,
               leaf_font_size=10,
               show_contracted=True,
               show_leaf_counts=True,color_threshold=50, above_threshold_color='k',no_labels =True) # Evidence to retain 3 or 4 clusters
    
    # Apply Hierarchical Clustering:
    hClustering = AgglomerativeClustering(n_clusters = nHC,
                                          affinity = 'euclidean',
                                          linkage = 'ward').fit(somHCCentroidsEngage)
    
    labelsSomHC2 = pd.DataFrame(hClustering.labels_)
    labelsSomHC2.columns =  ['labelsSomHC2']
    labelsSomHC2.reset_index(level=0, inplace=True)
    labelsSomHC2=labelsSomHC2.rename(columns={'index':'labelsSomHC'})
    
    # Join the new clusters to the data frame dfKmeansHC with merge through the 'LabelsKmeansHC'
    dfSomHC=dfSomHC.merge(labelsSomHC2, left_on='labelsSomHC', right_on='labelsSomHC')
 
    # Box Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (Som + Hierarchical)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.boxplot(x='labelsSomHC2', y=i, data=dfSomHC)
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Violin Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (K-means + Hierarchical)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.violinplot(x='labelsSomHC2', y=i, data=dfSomHC, scale='width')
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Silhouette Graph:   
    silhouette_avg = silhouette_score(dfNorm,dfSomHC['labelsSomHC2']) 
    print("For number of cluster (k) =", nHC,
              "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dfNorm,dfSomHC['labelsSomHC2'])
    cluster_labels = dfSomHC['labelsSomHC2']
    y_lower = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, dfNorm.shape[0] + (nHC + 1) * 10]) 
    
    for i in range(nHC):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()    
        size_cluster_i=ith_cluster_silhouette_values. shape[0]
        y_upper = y_lower + size_cluster_i    
        color = cm.nipy_spectral(float(i) / nHC)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color,
                              edgecolor=color, 
                              alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        ax.set_title("Silhouette Plot")
        ax.set_xlabel("The Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()
    
    if (returndf==True):
        return dfSomHC 
    
#####################################    
# EM
#####################################  
def EM_funct(n, dfNorm, dfEM,returndf=False):
    gmm = mixture.GaussianMixture(n_components = n, 
                                  init_params='kmeans', # {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
                                  max_iter=1000,
                                  n_init=20,
                                  verbose = 1).fit(dfNorm)
    
    labelsEm = pd.DataFrame(gmm.predict(dfNorm))
    labelsEm.columns = ['labelsEm']
    dfEM = pd.DataFrame(pd.concat([dfEM, labelsEm],axis=1))
    # Box Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (EM)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.boxplot(x='labelsEm', y=i, data=dfEM)
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Violin Plots:
    lista=list(dfNorm.columns)
    fig2=plt.figure()
    fig2.suptitle('Violin Plots by Variable and Cluster (EM)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.violinplot(x='labelsEm', y=i, data=dfEM, scale='width')
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    #Silhouette Graph:
    silhouette_avg = silhouette_score(dfNorm, gmm.predict(dfNorm)) 
    print("For number of cluster (k) =", n,
              "The average silhouette_score is :", silhouette_avg)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dfNorm, gmm.predict(dfNorm))
    cluster_labels = gmm.predict(dfNorm)
    y_lower = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, dfNorm.shape[0] + (n + 1) * 10]) 
    
    for i in range(n):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()    
        size_cluster_i=ith_cluster_silhouette_values. shape[0]
        y_upper = y_lower + size_cluster_i    
        color = cm.nipy_spectral(float(i) / n)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color,
                              edgecolor=color, 
                              alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        ax.set_title("Silhouette Plot")
        ax.set_xlabel("The Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()
    
    if (returndf==True):
        return dfEM 
    
#####################################    
# Mean- Shift
#####################################     
def MeanShift_funct(dfNorm,dfMeanShift,returndf=False):
    my_bandwidth = estimate_bandwidth(dfNorm,
                                      quantile=0.3,
                                      n_samples=2000)
    
    ms = MeanShift(bandwidth=my_bandwidth,
                   bin_seeding=True).fit(dfNorm)
    
    labelsMs = pd.DataFrame(ms.labels_)
    labelsMs.columns = ['labelsMs']

    dfMeanShift = pd.DataFrame(pd.concat([dfMeanShift, labelsMs],axis=1))
        
    # Box Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (Mean Shift)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.boxplot(x='labelsMs', y=i, data=dfMeanShift)
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Violin Plots:
    lista=list(dfNorm.columns)
    fig2=plt.figure()
    fig2.suptitle('Violin Plots by Variable and Cluster (Mean Shift)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.violinplot(x='labelsMs', y=i, data=dfMeanShift, scale='width')
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    if (returndf==True):
        return dfMeanShift 

#####################################    
# DB Scan:
#####################################        


def DBScan_funct(dfNorm, dfDB,returndf=False):
    db = DBSCAN(eps= 0.75, #radius (euclidean distance)
                min_samples=10).fit(dfNorm) # minimum number of points inside the radius.
    labelsDB = pd.DataFrame(db.labels_)
    labelsDB.columns = ['labelsDB']
    dfDB = pd.DataFrame(pd.concat([dfDB, labelsDB],axis=1))
    # Box Plots:
    lista=list(dfNorm.columns)
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (DB-Scan)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.boxplot(x='labelsDB', y=i, data=dfDB)
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Violin Plots:
    lista=list(dfNorm.columns)
    fig2=plt.figure()
    fig2.suptitle('Violin Plots by Variable and Cluster (DB-Scan)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.violinplot(x='labelsDB', y=i, data=dfDB, scale='width')
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    if (returndf==True):
        return dfDB 

#####################################    
# K-Modes:
#####################################  
def Kmodes_funct(n,dfKmodesChange,dfKmodes,returndf=False):
    km = KModes(n_clusters=n, init='random', n_init=50, verbose=1) 
    clusters = km.fit_predict(dfKmodesChange)
    labelsKmodes=pd.DataFrame(km.labels_)
    labelsKmodes.columns = ['labelsKmodes']
    
    dfKmodes = pd.DataFrame(pd.concat([dfKmodes, labelsKmodes],axis=1))
    #Visualization:
    lista=list(dfKmodesChange.columns)
    for i in lista:
        g = sb.catplot(i, col="labelsKmodes", col_wrap=4,
                        data=dfKmodes,
                        kind="count", height=3.5, aspect=0.8, 
                        palette='tab20')
    plt.plot()   
    if (returndf==True):
        return dfKmodes 
###########################################################################################################

# 2 groups of variables: Value/ Engage (costumers) and Consumption/ Affinity (products)
dfEngage = dfWork[[  'YearsWus1998',
                      'salary',
                      'CMV_Mean_corrected',
                      'ratioSalaryLOB'
                      #'claims' # use claims only when ploting for comparing clusters
                      ]] 
dfEngageCat = dfWork[['catClaims',
                      'binEducation',
                      'children',
                      'CancelTotal',
                      'TotalInsurance'
                         #'catCMV' #TODO: Implement
                         ]]
dfEngageCat2 = dfWork[['catClaims',
                      'education',
                      'children',
                      'CancelTotal',
                      'TotalInsurance'
                         #'catCMV' #TODO: Implement
                         ]]

dfAffinity= dfWork[['lobMotor',
                  'lobHousehold',
                  'lobHealth',
                  'lobLife',
                  'lobWork']]

dfAffinityRatio=dfWork[['lobTotal',
                      'motorRatioLOB',
                      'householdRatioLOB',
                      'healthRatioLOB',
                      'lifeRatioLOB',
                      'workCRatioLOB']]
#################################################################################################################
#PEARSON CORRELATIONS: to check correlation between variables on each variable group.
#################################################################################################################

# Check correlations between variables in dfEngage:
plt.figure()
dfCorr=pd.DataFrame(dfWork,columns=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB'])
dfCorrP=dfCorr.corr(method ='pearson')
ax = sb.heatmap(dfCorrP, annot=True, fmt="0.3", square=True,  cbar_kws={'label': 'Colorbar'})
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.title('Heatmap Pearson Correlations (Group 1: Engage)')
plt.plot()
# Non highly correlated variables

# Check correlations between variables in dfAffinity:
plt.figure()
dfCorr=pd.DataFrame(dfWork,columns=['lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
dfCorrP=dfCorr.corr(method ='pearson')
ax = sb.heatmap(dfCorrP, annot=True, fmt="0.3", square=True,  cbar_kws={'label': 'Colorbar'})
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.title('Heatmap Pearson Correlations (Group 3: Affinity)')
plt.plot()
# Non highly correlated variables

# Check correlations between variables in dfAffinityRatio:
plt.figure()
dfCorr=pd.DataFrame(dfWork,columns=['lobTotal','motorRatioLOB','householdRatioLOB','healthRatioLOB','lifeRatioLOB','workCRatioLOB'])
dfCorrP=dfCorr.corr(method ='pearson')
ax = sb.heatmap(dfCorrP, annot=True, fmt="0.3", square=True,  cbar_kws={'label': 'Colorbar'})
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.title('Heatmap Pearson Correlations (Group 1: Affinity Ratio)')
plt.plot()
# There is a correlation between householdRatioLOB and lobTotal 
# Multicolinearity if we keep all ratio LOB variables in the same group of variables.
# Decision: drop householdRatioLOB from dfAffinityRatio.
##############################################################################################
# Apply Clusters: dfEngage
########################################### K-means ##########################################
# Normalization: min-max
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
dfEngageKmeans=dfEngage.copy()
engageNorm = scaler.fit_transform(dfEngageKmeans)
engageNorm = pd.DataFrame(engageNorm, columns = dfEngageKmeans.columns)
engageNorm = engageNorm.rename(columns={'YearsWus1998':'YearsWus1998Std','salary':'salaryStd','CMV_Mean_corrected':'CMV_Mean_correctedStd','ratioSalaryLOB':'ratioSalaryLOBStd'}, errors="raise")
dfEngageKmeans=pd.DataFrame(pd.concat([dfEngageKmeans, engageNorm],axis=1), 
                        columns=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageKmeans=pd.DataFrame(pd.concat([dfWork['id'], dfEngageKmeans],axis=1), 
                        columns=['id','YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])

# Elbow Graph: to check how many clusters we should have:
cluster_range= range(1,7)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(n_clusters=num_clusters, 
                        random_state=0,
                        n_init = 20, 
                        max_iter = 300,
                        init='k-means++')
    clusters.fit(engageNorm)
    cluster_errors.append(clusters.inertia_)

dfClusters = pd.DataFrame({ "Num_clusters": cluster_range, "Cluster_errors": cluster_errors})
plt.figure(figsize=(1,8))
plt.xlabel("Cluster Number")
plt.ylabel("Within- Cluster Sum of Squares")
plt.title('Elbow Graph')
plt.plot(dfClusters.Num_clusters,dfClusters.Cluster_errors,marker='o') # There is evidence that we should keep 2 clusters.

# k-means with 2 clusters:
dfEngageKmeans=kmeans_funct(dfKmeans=dfEngageKmeans,dfNorm=engageNorm, n=2,returndf=True)

# clusters only differentiate on 'YearsWus1998'

# We also tried to make 3 clusters instead to check if with 3 clusters we could differentiate according to salary and cmv.
#dfEngageKmeans=kmeans_funct(dfKmeans=dfEngageKmeans,dfNorm=engageNorm, n=3,returndf=False)

#TODO: Ver K-means e K-modes ao mesmo tempo:

# Violin Plots:
#    lista=list(engageNorm.columns)
#    fig2=plt.figure()
#    fig2.suptitle('Violin Plots by Variable and Cluster (labels K-means and variables k-modes)')
#    for i in lista:
#        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
#        sb.violinplot(x='labelsKmeans', y=i, data=dfEngageKmeans, scale='width')
#        plt.xlabel('Cluster Number')
#        plt.title(str(i))
#        plt.ylabel('')
#    plt.tight_layout()
#    plt.plot()


########################################### K-means + Hierarchical ##########################################
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

dfEngageKmeansHC=dfEngage
dfEngageKmeansHC=pd.DataFrame(pd.concat([dfEngageKmeansHC, engageNorm],axis=1), 
                        columns=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageKmeansHC=pd.DataFrame(pd.concat([dfWork['id'], dfEngageKmeansHC],axis=1), 
                        columns=['id','YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
#dfEngageKmeansHC=kmeansHC_funct(dfNorm=engageNorm,dfKmeansHC=dfEngageKmeansHC,nkmeans=int(round(math.sqrt(10261),0)),nHC=3,returndf=True)
dfEngageKmeansHC=kmeansHC_funct(dfNorm=engageNorm,dfKmeansHC=dfEngageKmeansHC,nkmeans=int(round(math.sqrt(10261),0)),nHC=2,returndf=True)

########################################### SOM + Hierarchical ##########################################
from sompy.sompy import SOMFactory
set_seed(my_seed)
dfEngageSomHC=dfEngage
dfEngageSomHC=pd.DataFrame(pd.concat([dfEngageSomHC, engageNorm],axis=1), 
                        columns=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageSomHC=pd.DataFrame(pd.concat([dfWork['id'], dfEngageSomHC],axis=1), 
                        columns=['id','YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
names = ['YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd']

dfEngageSomHC=SomHC_funct(dfNorm=engageNorm,dfSomHC=dfEngageSomHC,names=names,nHC=4,returndf=True)

########################################### EM ##########################################
from sklearn import mixture
set_seed(my_seed)
dfEngageEM=dfEngage
dfEngageEM=pd.DataFrame(pd.concat([dfEngageEM, engageNorm],axis=1), 
                        columns=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageEM=pd.DataFrame(pd.concat([dfWork['id'], dfEngageEM],axis=1), 
                        columns=['id','YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageEM=EM_funct(dfNorm=engageNorm, dfEM=dfEngageEM, n=3,returndf=True)
########################################### Mean Shift ##########################################
from sklearn.cluster import MeanShift, estimate_bandwidth
set_seed(my_seed)
dfEngageMs=dfEngage
dfEngageMs=pd.DataFrame(pd.concat([dfEngageMs, engageNorm],axis=1), 
                        columns=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageMs=pd.DataFrame(pd.concat([dfWork['id'], dfEngageMs],axis=1), 
                        columns=['id','YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])

dfEngageMs=MeanShift_funct(dfNorm=engageNorm,dfMeanShift=dfEngageMs,returndf=True)

########################################### DB- Scan ##########################################
# Esquecer isto porque não faz sentido. As densidades são muito similares.
from sklearn.cluster import DBSCAN
dfEngageDb=dfEngage
dfEngageDb=pd.DataFrame(pd.concat([dfEngageDb, engageNorm],axis=1), 
                        columns=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageDb=pd.DataFrame(pd.concat([dfWork['id'], dfEngageDb],axis=1), 
                        columns=['id','YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','YearsWus1998Std','salaryStd','CMV_Mean_correctedStd','ratioSalaryLOBStd'])
dfEngageDb=DBScan_funct(dfNorm=engageNorm,dfDB=dfEngageDb,returndf=True)

##############################################################################################
# Apply Clusters: dfEngageCat
########################################### K-modes binEduc: ##########################################
from kmodes.kmodes import KModes
set_seed(my_seed)
dfEngageCatKmodes=dfEngageCat
dfEngageCatKmodes=pd.DataFrame(pd.concat([dfWork['id'], dfEngageCatKmodes],axis=1), 
                        columns=['id','catClaims','binEducation','children','CancelTotal','TotalInsurance'])
kmodesChange = dfEngageCatKmodes[['catClaims','binEducation','children','CancelTotal','TotalInsurance']].astype('str')

dfEngageCatKmodes=Kmodes_funct(n=2,dfKmodesChange=kmodesChange,dfKmodes=dfEngageCatKmodes,returndf=True)

########################################### K-modes education: ##########################################
from kmodes.kmodes import KModes
set_seed(my_seed)
dfEngageCatKmodes2=dfEngageCat2.copy()
dfEngageCatKmodes2=pd.DataFrame(pd.concat([dfWork['id'], dfEngageCatKmodes2],axis=1), 
                        columns=['id','catClaims','education','children','CancelTotal','TotalInsurance'])
kmodesChange2 = dfEngageCatKmodes2[['catClaims','education','children','CancelTotal','TotalInsurance']].astype('str')

dfEngageCatKmodes2=Kmodes_funct(n=2,dfKmodesChange=kmodesChange2,dfKmodes=dfEngageCatKmodes2,returndf=True)
##############################################################################################
# Apply Clusters: dfAffinity
########################################### K-means ##########################################
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
set_seed(my_seed)
dfAffinityKmeans=dfAffinity
dfAffinityKmeansNStd=dfAffinity
dfAffinityKmeans=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityKmeans],axis=1), 
                        columns=['id','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])

# Elbow Graph: to check how many clusters we should have:
cluster_range= range(1,7)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(n_clusters=num_clusters, 
                        random_state=0,
                        n_init = 20, 
                        max_iter = 300,
                        init='k-means++')
    clusters.fit(dfAffinityKmeansNStd)
    cluster_errors.append(clusters.inertia_)

dfClusters = pd.DataFrame({ "Num_clusters": cluster_range, "Cluster_errors": cluster_errors})
plt.figure(figsize=(1,8))
plt.xlabel("Cluster Number")
plt.ylabel("Within- Cluster Sum of Squares")
plt.title('Elbow Graph')
plt.plot(dfClusters.Num_clusters,dfClusters.Cluster_errors,marker='o') # There is evidence that we should keep 2 clusters.

# k-means with 2 clusters:
dfAffinityKmeans=kmeans_funct(dfKmeans=dfAffinityKmeans,dfNorm=dfAffinityKmeansNStd, n=3,returndf=False)
# clusters only differentiate on 'YearsWus1998'
# We also tried to make 3 clusters instead to check if with 3 clusters we could differentiate according to salary and cmv.
#dfEngageKmeans=kmeans_funct(dfKmeans=dfEngageKmeans,dfNorm=engageNorm, n=3,returndf=False)

########################################### K-means + Hierarchical ##########################################
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
set_seed(my_seed)
dfAffinityKmeansHC=dfAffinity
dfAffinityKmeansHCNStd=dfAffinity
dfAffinityKmeansHC=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityKmeansHC],axis=1), 
                        columns=['id','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
#dfEngageKmeansHC=kmeansHC_funct(dfNorm=engageNorm,dfKmeansHC=dfEngageKmeansHC,nkmeans=int(round(math.sqrt(10261),0)),nHC=3,returndf=True)
dfAffinityKmeansHC=kmeansHC_funct(dfNorm=dfAffinityKmeansHCNStd,dfKmeansHC=dfAffinityKmeansHC,nkmeans=int(round(math.sqrt(10261),0)),nHC=3,returndf=True)
########################################### SOM + Hierarchical ##########################################
from sompy.sompy import SOMFactory
set_seed(my_seed)
dfAffinitySomHC=dfAffinity
dfAffinitySomHCNStd=dfAffinity
dfAffinitySomHC=pd.DataFrame(pd.concat([dfWork['id'], dfAffinitySomHC],axis=1), 
                        columns=['id','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
names = ['lobMotor','lobHousehold','lobHealth','lobLife','lobWork']

dfAffinitySomHC=SomHC_funct(dfNorm=dfAffinitySomHCNStd,dfSomHC=dfAffinitySomHC,names=names,nHC=3,returndf=True)

########################################### EM ##########################################
from sklearn import mixture
set_seed(my_seed)
dfAffinityEM=dfAffinity
dfAffinityEMNStd=dfAffinity
dfAffinityEM=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityEM],axis=1), 
                        columns=['id','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
dfAffinityEM=EM_funct(dfNorm=dfAffinityEMNStd, dfEM=dfAffinityEM, n=3,returndf=True)

#Not used yet:
# Likelihood value
#EM_score_samp = pd.DataFrame(gmm.score_samples(engageNorm))
# Probabilities of belonging to each cluster.
#EM_pred_prob = pd.DataFrame(gmm.predict_proba(engageNorm))

########################################### Mean Shift ##########################################
from sklearn.cluster import MeanShift, estimate_bandwidth
set_seed(my_seed)
dfAffinityMs=dfAffinity
dfAffinityMsNStd=dfAffinity
dfAffinityMs=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityMs],axis=1), 
                        columns=['id','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])

dfAffinityMs=MeanShift_funct(dfNorm=dfAffinityMsNStd,dfMeanShift=dfAffinityMs,returndf=True)

########################################### DB- Scan ##########################################
# Esquecer isto porque não faz sentido. As densidades são muito similares.
from sklearn.cluster import DBSCAN
set_seed(my_seed)
dfAffinityDb=dfAffinity
dfAffinityDbNStd=dfAffinity
dfAffinityDb=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityDb],axis=1), 
                        columns=['id','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
dfAffinityDb=DBScan_funct(dfNorm=dfAffinityDbNStd,dfDB=dfAffinityDb,returndf=True)



##############################################################################################
# Apply Clusters: dfAffinityRatio
########################################### K-means ##########################################
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
set_seed(my_seed)
dfAffinityRatioKmeans=dfAffinityRatio.copy()
AffinityRatioNorm=dfAffinityRatio.copy()
AffinityRatioNorm['lobTotalStd']=minmax_scale(AffinityRatioNorm[['lobTotal']])
AffinityRatioNorm=AffinityRatioNorm.drop(columns=['lobTotal'])
dfAffinityRatioKmeans=pd.DataFrame(pd.concat([dfAffinityRatioKmeans, AffinityRatioNorm['lobTotalStd']],axis=1))
dfAffinityRatioKmeans=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityRatioKmeans],axis=1))

# Elbow Graph: to check how many clusters we should have:
cluster_range= range(1,7)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(n_clusters=num_clusters, 
                        random_state=0,
                        n_init = 20, 
                        max_iter = 300,
                        init='k-means++')
    clusters.fit(AffinityRatioNorm)
    cluster_errors.append(clusters.inertia_)

dfClusters = pd.DataFrame({ "Num_clusters": cluster_range, "Cluster_errors": cluster_errors})
plt.figure(figsize=(1,8))
plt.xlabel("Cluster Number")
plt.ylabel("Within- Cluster Sum of Squares")
plt.title('Elbow Graph')
plt.plot(dfClusters.Num_clusters,dfClusters.Cluster_errors,marker='o') # There is evidence that we should keep 2 clusters.

# k-means with 2 clusters:
dfAffinityRatioKmeans=kmeans_funct(dfKmeans=dfAffinityRatioKmeans,dfNorm=AffinityRatioNorm, n=3,returndf=True)
# clusters only differentiate on 'YearsWus1998'
# We also tried to make 3 clusters instead to check if with 3 clusters we could differentiate according to salary and cmv.
#dfEngageKmeans=kmeans_funct(dfKmeans=dfEngageKmeans,dfNorm=engageNorm, n=3,returndf=False)

########################################### K-means + Hierarchical ##########################################
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
set_seed(my_seed)
dfAffinityRatioKmeansHC=dfAffinityRatio.copy()
dfAffinityRatioKmeansHC=pd.DataFrame(pd.concat([dfAffinityRatioKmeansHC, AffinityRatioNorm['lobTotalStd']],axis=1))
dfAffinityRatioKmeansHC=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityRatioKmeansHC],axis=1))
#dfEngageKmeansHC=kmeansHC_funct(dfNorm=engageNorm,dfKmeansHC=dfEngageKmeansHC,nkmeans=int(round(math.sqrt(10261),0)),nHC=3,returndf=True)
dfAffinityRatioKmeansHC=kmeansHC_funct(dfNorm=AffinityRatioNorm,dfKmeansHC=dfAffinityRatioKmeansHC,nkmeans=int(round(math.sqrt(10261),0)),nHC=3,returndf=True)
########################################### SOM + Hierarchical ##########################################
from sompy.sompy import SOMFactory
set_seed(my_seed)
dfAffinityRatioSomHC=dfAffinityRatio.copy()
dfAffinityRatioSomHC=pd.DataFrame(pd.concat([dfAffinityRatioSomHC, AffinityRatioNorm['lobTotalStd']],axis=1))
dfAffinityRatioSomHC=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityRatioSomHC],axis=1))
names = ['lobTotal','motorRatioLOB','householdRatioLOB','healthRatioLOB','lifeRatioLOB','workCRatioLOB']

dfAffinityRatioSomHC=SomHC_funct(dfNorm=AffinityRatioNorm,dfSomHC=dfAffinityRatioSomHC,names=names,nHC=4,returndf=True)

########################################### EM ##########################################
from sklearn import mixture
set_seed(my_seed)
dfAffinityRatioEM=dfAffinityRatio.copy()
dfAffinityRatioEM=pd.DataFrame(pd.concat([dfAffinityRatioEM, AffinityRatioNorm['lobTotalStd']],axis=1))
dfAffinityRatioEM=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityRatioEM],axis=1))
dfAffinityRatioEM=EM_funct(dfNorm=AffinityRatioNorm, dfEM=dfAffinityRatioEM, n=3,returndf=True)

# Not used yet:
# Likelihood value
# EM_score_samp = pd.DataFrame(gmm.score_samples(engageNorm))
# Probabilities of belonging to each cluster.
# EM_pred_prob = pd.DataFrame(gmm.predict_proba(engageNorm))
########################################### Mean Shift ##########################################
from sklearn.cluster import MeanShift, estimate_bandwidth
set_seed(my_seed)
dfAffinityRatioMs=dfAffinityRatio.copy()
dfAffinityRatioMs=pd.DataFrame(pd.concat([dfAffinityRatioMs, AffinityRatioNorm['lobTotalStd']],axis=1))
dfAffinityRatioMs=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityRatioMs],axis=1))
dfAffinityRatioMs=MeanShift_funct(dfNorm=AffinityRatioNorm,dfMeanShift=dfAffinityRatioMs,returndf=True)

########################################### DB- Scan ##########################################
# Esquecer isto porque não faz sentido. As densidades são muito similares.
from sklearn.cluster import DBSCAN
set_seed(my_seed)
dfAffinityRatioDb=dfAffinityRatio.copy()
dfAffinityRatioDb=pd.DataFrame(pd.concat([dfAffinityRatioDb, AffinityRatioNorm['lobTotalStd']],axis=1))
dfAffinityRatioDb=pd.DataFrame(pd.concat([dfWork['id'], dfAffinityRatioDb],axis=1))
dfAffinityRatioDb=DBScan_funct(dfNorm=AffinityRatioNorm,dfDB=dfAffinityRatioDb,returndf=True)


##############################################################################################
#INSERT LABELS IN dfWork:
##############################################################################################
dfWork= pd.DataFrame(pd.concat([dfWork,dfEngageMs['labelsMs']],axis=1))
dfWork= pd.DataFrame(pd.concat([dfWork,dfEngageCatKmodes2['labelsKmodes']],axis=1))
dfWork= pd.DataFrame(pd.concat([dfWork,dfAffinityRatioKmeansHC['labelsKmeansHC2']],axis=1))
dfWork['labelsConcat']=dfWork['labelsMs'].astype(str)+dfWork['labelsKmodes'].astype(str)+dfWork['labelsKmeansHC2'].astype(str)

############################################
##DECISION TREE
############################################
#import numpy as np
#import pandas as pd
#from sklearn import preprocessing
#from sklearn.model_selection import cross_val_score
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier, plot_tree
#from sklearn.model_selection import train_test_split # Import train_test_split function
#from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
##from dtreeplt import dtreeplt
#import graphviz 
#
#values, counts= np.unique(dfWork['labelsConcat'], return_counts=True)
#pd.DataFrame(np.asarray((values, counts)).T, columns=['labelsConcat','Number'])
#print(values,counts)
#
#le = preprocessing.LabelEncoder()
#clf = DecisionTreeClassifier(random_state=0,
#                             max_depth=3, dtype=category) # define the depth of the decision tree!
##dfWork2=dfWork.copy()
##dfWork2['catClaims']=dfWork2['catClaims'].astype(str)
##dfWork2['education']=dfWork2['education'].astype(str)
##dfWork2['children']=dfWork2['children'].astype(str)
##dfWork2['CancelTotal']=dfWork2['CancelTotal'].astype(str)
##dfWork2['TotalInsurance']=dfWork2['TotalInsurance'].astype(str)
#
#X = dfWork2[['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB',
#              'catClaims','education','children','CancelTotal',
#              'TotalInsurance','lobTotal','motorRatioLOB','householdRatioLOB','healthRatioLOB','lifeRatioLOB','workCRatioLOB']]
#
#y =  dfWork2[['labelsConcat']] # Target variable
#
## How many elements per Cluster
## Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                    test_size=4, 
#                                                    random_state=1) # 70% training and 30% tes
#
## Create Decision Tree classifer object
##clf = DecisionTreeClassifier()
#
## Train Decision Tree Classifer
#clf = clf.fit(X_train,y_train)
#
#clf.feature_importances_
#
##plot_tree(clf, filled=True)
#
##Entropy or gini: to see which variables are more relevant - avoid overfitting (the most imp. variable will be on the top of the DT)
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#
##Label tree: instead of X[0] put the names of the variables
#dot_data = tree.export_graphviz(clf, out_file=None,
#                                feature_names=X_train.columns,
#                                class_names = X_train.columns,
#                                filled=True,
#                                rounded=True,
#                                special_characters=True)  
#graph = graphviz.Source(dot_data)
#graph
#
#to_class = {'clothes':[99,10,5, 0],
#        'kitchen':[1, 60, 5, 90],
#        'small_appliances':[0, 5, 75, 2],
#        'toys':[0,5, 5, 7],
#        'house_keeping':[0, 20, 10, 1]}
#
## Creates pandas DataFrame. 
#to_class = pd.DataFrame(to_class, index =['cust1', 'cust2', 'cust3', 'cust4']) 
#to_class['label']=clf.predict(to_class)
## Classify these new elements

###########################################################################
# Check numbers of obs in each cluster (concat)
dfWork['labelsConcat'].value_counts()
# There are clusters with few obs like 1, 23, 18 and 11 observation clusters.
# Do violin plots for the clusters that have 23, 18 and 11 individuals (clusters: 110,111,112)

lista=list(dfWork.columns)
lista1=['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','lobMotor',
 'lobHousehold',
 'lobHealth']
lista2=[
 'lobLife',
 'lobWork','lobTotal','motorRatioLOB',
 'healthRatioLOB',
 'lifeRatioLOB',
 'workCRatioLOB']
df=dfWork[(dfWork['labelsConcatKNN']=='110')|(dfWork['labelsConcatKNN']=='111')|(dfWork['labelsConcatKNN']=='112')]

fig=plt.figure()
fig.suptitle('Violin Plots by Variable and Cluster (Concat Clusters)')
for i in lista1:
    plt.subplot2grid((1,len(lista1)),(0,lista1.index(i)))
    sb.violinplot(x='labelsConcat', y=i, data=df, scale='width')
    plt.xlabel('Cluster Number',fontsize=8)
    plt.title(str(i), fontsize=8)
    plt.ylabel('')
plt.tight_layout()
plt.plot()

fig=plt.figure()
fig.suptitle('Violin Plots by Variable and Cluster (Concat Clusters)')
for i in lista2:
    plt.subplot2grid((1,len(lista2)),(0,lista2.index(i)))
    sb.violinplot(x='labelsConcat', y=i, data=df, scale='width')
    plt.xlabel('Cluster Number',fontsize=8)
    plt.title(str(i), fontsize=8)
    plt.ylabel('')
plt.tight_layout()
plt.plot()

# Lets check the same plots for the clusters that have more observations:

lista=list(dfWork.columns)
lista1=['YearsWus1998','salary','CMV_Mean_corrected']
lista2=['ratioSalaryLOB','lobMotor',
 'lobHousehold',
 'lobHealth']
lista3=[
 'lobLife',
 'lobWork','lobTotal']
lista4=['motorRatioLOB',
 'healthRatioLOB',
 'lifeRatioLOB',
 'workCRatioLOB']

dfbig=dfWork[(dfWork['labelsConcatKNN']!='110')&(dfWork['labelsConcatKNN']!='111')&(dfWork['labelsConcatKNN']!='112')&(dfWork['labelsConcatKNN']!='100')&(dfWork['labelsConcatKNN']!='101')]
fig=plt.figure()
fig.suptitle('Violin Plots by Variable and Cluster (Concat Clusters)')
for i in lista1:
    plt.subplot2grid((1,len(lista1)),(0,lista1.index(i)))
    sb.violinplot(x='labelsConcat', y=i, data=dfbig, scale='width')
    plt.xlabel('Cluster Number',fontsize=8)
    plt.title(str(i), fontsize=8)
    plt.ylabel('')
plt.tight_layout()
plt.plot()

fig=plt.figure()
fig.suptitle('Violin Plots by Variable and Cluster (Concat Clusters)')
for i in lista2:
    plt.subplot2grid((1,len(lista2)),(0,lista2.index(i)))
    sb.violinplot(x='labelsConcat', y=i, data=dfbig, scale='width')
    plt.xlabel('Cluster Number',fontsize=8)
    plt.title(str(i), fontsize=8)
    plt.ylabel('')
plt.tight_layout()
plt.plot()

fig=plt.figure()
fig.suptitle('Violin Plots by Variable and Cluster (Concat Clusters)')
for i in lista3:
    plt.subplot2grid((1,len(lista3)),(0,lista3.index(i)))
    sb.violinplot(x='labelsConcat', y=i, data=dfbig, scale='width')
    plt.xlabel('Cluster Number',fontsize=8)
    plt.title(str(i), fontsize=8)
    plt.ylabel('')
plt.tight_layout()
plt.plot()

fig=plt.figure()
fig.suptitle('Violin Plots by Variable and Cluster (Concat Clusters)')
for i in lista4:
    plt.subplot2grid((1,len(lista4)),(0,lista4.index(i)))
    sb.violinplot(x='labelsConcat', y=i, data=dfbig, scale='width')
    plt.xlabel('Cluster Number',fontsize=8)
    plt.title(str(i), fontsize=8)
    plt.ylabel('')
plt.tight_layout()
plt.plot()

#Histograms
# 3 Smaller not counting with clusters with 1 obs:
df=dfWork[(dfWork['labelsConcat']=='110')|(dfWork['labelsConcat']=='111')|(dfWork['labelsConcat']=='112')]
Hist=dfWork[['catClaims','education','children','CancelTotal','TotalInsurance']].astype('str')
lista=list(Hist.columns)

for i in lista:
    g = sb.catplot(i, col="labelsConcat", col_wrap=4,
                    data=df,
                    kind="count", height=3.5, aspect=0.8, 
                    palette='tab20')
plt.plot()

# 3 bigger clusters:
dfbig=dfWork[(dfWork['labelsConcat']!='110')&(dfWork['labelsConcat']!='111')&(dfWork['labelsConcat']!='112')&(dfWork['labelsConcat']!='100')&(dfWork['labelsConcat']!='101')]
for i in lista:
    g = sb.catplot(i, col="labelsConcat", col_wrap=4,
                    data=dfbig,
                    kind="count", height=3.5, aspect=0.8, 
                    palette='tab20')
plt.plot()   

###########################################################################
# Join the smaller clusters that have the first number 1 in one unique cluster called 1:
dfWork['Finallabels']=np.where((dfWork['labelsConcat']=='110')|(dfWork['labelsConcat']=='111')|(dfWork['labelsConcat']=='112')|(dfWork['labelsConcat']=='100')|(dfWork['labelsConcat']=='101'),'3','2')
dfWork['Finallabels']=np.where((dfWork['labelsConcat']=='011')|(dfWork['labelsConcat']=='010')|(dfWork['labelsConcat']=='012'),'1',dfWork['Finallabels'])
##########################################################################
#Delete columns from dfOutliers that were also deleted from dfWork
dfOutliers=dfOutliers.drop(columns=['birthday','Strange_birthday','strange_firstPolicy','Outliers_salary','df_OutliersLow_cmv','df_OutliersHigh_cmv','Outliers_lobMot','Outliers_lobHousehold','Outliers_lobHealth','Outliers_lobWork','age','age1998','incoherences'])

#Impute Outliers:
dfWorkTreat=dfWork.copy()
dfWorkTreat=dfWorkTreat.drop(columns=['labelsMs','labelsKmodes','labelsKmeansHC2','labelsConcat'])
dfOutliers['Finallabels']=np.NaN

dfWorkTreat=pd.DataFrame(pd.concat([dfWorkTreat, dfOutliers],axis=0))

#Impute Outliers:
dfImputeOut=dfWorkTreat[['id','YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','lobMotor',
        'lobHousehold','lobHealth','lobLife','lobWork','lobTotal','motorRatioLOB',
        'healthRatioLOB','lifeRatioLOB','workCRatioLOB','Finallabels']]
dfImputeOutvar=dfWorkTreat[['YearsWus1998','salary','CMV_Mean_corrected','ratioSalaryLOB','lobMotor',
        'lobHousehold','lobHealth','lobLife','lobWork','lobTotal','motorRatioLOB',
        'healthRatioLOB','lifeRatioLOB','workCRatioLOB','Finallabels']]

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance")

dfWorkTreat['Finallabels2']=imputer.fit_transform(dfImputeOutvar)[:,[14]]
dfWorkTreat['Finallabels2']= round(dfWorkTreat['Finallabels2'],0)
dfWorkTreat=pd.merge(dfWorkTreat,dfOriginal[['id','Others']],on='id')
Outliers=dfWorkTreat[['id','Finallabels2']][dfWorkTreat['Others']>0]
All=dfWorkTreat[['id','Finallabels2']]

Outliers.to_excel(f"./Outliers.xlsx", index=False, encoding='utf-8')
All.to_excel(f"./FinalData.xlsx", index=False, encoding='utf-8')
