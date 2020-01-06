# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:44:55 2019

@author: aSUS
"""
#IMPORTS
import pandas as pd
#!pip install modin[dask]
#import modin.pandas as pd # replaces pandas for parallel running, defaults to pandas when better method not available

import sqlite3
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

#my_path = 'C:/Users/aSUS/Documents/IMS/Master Data Science and Advanced Analytics with major in BA/Data mining/Projeto/insurance.db'
#my_path = r'C:\Users\Pedro\Google Drive\IMS\1S-Master\Data Mining\Projecto\insurance.db'
##dbname = "datamining.db"
#
## connect to the database
#conn = sqlite3.connect(my_path)
##cursor = conn.cursor()
#conn.row_factory=sqlite3.Row
#
##tables in the data base:
#query = "select name from sqlite_master where type='table'"
#df2 = pd.read_sql_query(query,conn)
#
##Columns in each table of the data base:
#query2="select sql from sqlite_master where tbl_name='LOB' and type='table'"
#print(pd.read_sql_query(query2,conn))
#
#cur.execute('select * from LOB')
#col_name_list=[tuple[0] for tuple in cur.description]
#
#query="select * from lob limit(10);"
#
#query="select * from engage limit(10);"
#
#my_table= cursor.execute(query).fetchall()
#cursor.execute("select name from sqlite_master where type='table'")
#print(cursor.fetchall()) """
#FUNCTIONS
def set_seed(my_seed):
    random.seed(my_seed)
    np.random.seed(my_seed)
my_seed=100
#Diretorias:
#file='C:/Users/aSUS/Documents/IMS/Master Data Science and Advanced Analytics with major in BA/Data mining/Projeto/A2Z Insurance.csv'
file= r'C:\Users\Pedro\Google Drive\IMS\1S-Master\Data Mining\Projecto\A2Z Insurance.csv'
#file='C:/Users/anaso/Desktop/Faculdade/Mestrado/Data Mining/Projeto/A2Z Insurance.csv'

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
#Create columns for global flags with zeros (hack)
dfOriginal['Others'] = np.where(dfOriginal['birthday']<0, 1,0)
dfOriginal['Errors'] = np.where(dfOriginal['birthday']<0, 1,0)
dfOriginal['Outliers'] = np.where(dfOriginal['birthday']<0, 1,0)
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
dfOriginal['Others'] = np.where(dfOriginal['cmv']>2000, 1,dfOriginal['Others'])         #assign flag do not enter in model
dfOriginal['Outliers'] = np.where(dfOriginal['cmv']<(-500), 1,dfOriginal['Outliers'])       #assign flag outlier
dfOriginal['Outliers'] = np.where(dfOriginal['cmv']>2000, 1,dfOriginal['Outliers'])       #assign flag outlier


cmvValues = dfOriginal['cmv'][(dfOriginal['df_OutliersLow_cmv']==0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)].value_counts().sort_index()
cmvValues
#There are 3 higher values that will be considered as outliers.

# Change the values of the new positive outliers to 1 in the df_OutliersHigh_cmv column
dfOriginal['df_OutliersHigh_cmv'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,dfOriginal['df_OutliersHigh_cmv'])

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
dfOriginal['Outliers'] = np.where(dfOriginal['claims']>4, 1,dfOriginal['Outliers']) #assign flag outlier
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
dfOriginal['Outliers'] = np.where(dfOriginal['lobMotor']>750, 1,dfOriginal['Outliers'])     #assign flag outlier
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
dfOriginal['Outliers_lobHousehold']=np.where(dfOriginal['lobHousehold']>3000,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobHousehold']>3000,1,dfOriginal['Others'])
dfOriginal['Outliers'] = np.where(dfOriginal['lobHousehold']>3000, 1,dfOriginal['Outliers'])
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
dfOriginal['Outliers'] = np.where(dfOriginal['lobHealth']>550, 1,dfOriginal['Outliers'])
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
dfOriginal['Outliers'] = np.where(dfOriginal['lobWork']>400, 1,dfOriginal['Outliers'])
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
#Outliers & Errors
######################################################################################################

dfOriginal['Outliers'].value_counts()
dfOriginal['Errors'].value_counts()
df=dfOriginal[["Outliers_salary","df_OutliersLow_cmv","df_OutliersHigh_cmv","Outliers_claims",
                "Outliers_lobMot","Outliers_lobHousehold","Outliers_lobHealth"]].loc[dfOriginal["Outliers"]==1]
#TODO: do the same for errors , report on errors
# There is a high number of individuals with low work premiums values.
# Transformar em Logaritmo? (ver mais tarde)

#-----------------CHECK INCOHERENCES------------------#
# TODO: create a column for incoherences to verify if errors and outliers coincide with incoherences
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
dfOriginal['children'][dfOriginal['age']<=16].value_counts()
# There are 31 people who are younger or equal to 16 years old and that have children and 16 younger or equal to 16 years old and that do not have children.
# At this age, in normal situations, there should be zero people with children. Even if there were some cases in these situations, 31 is a huge number.

# Check if people with age <9 years old have basic education.
# Check if people with age <16 years old have high school education. This would not make sense: in normal circumstances, people with less than 16 years old have not completed the high school education yet.
# Check if people with age <20 years old have Bsc/Msc education. This would not make sense: in normal circumstances, people with less than 20 years old have not completed the Bsc/ Msc education yet.
# Check if people with age <25 years old have Phd education. This would not make sense: in normal circumstances, people with less than 25 years old have not completed the Phd education yet.
dfOriginal['education'][dfOriginal['age']<9].value_counts()
dfOriginal[dfOriginal['age']<9] # There are no people with less than 9 years old.
dfOriginal['education'][dfOriginal['age']<16].value_counts() # People with less than 16 years old only have basic education (12 people), which makes sense.
dfOriginal['education'][dfOriginal['age']<20].value_counts() # People with less than 20 years old only have basic education (262 people) and high school education (81 people).
dfOriginal['education'][dfOriginal['age']<25].value_counts() #There are 8 people who have a Phd with age less than 25 years old, which does not make sense.

# Check if people with less than 16 years old have a salary>0
dfOriginal['salary'][(dfOriginal['age']<16) & (dfOriginal['salary']>0)].count() #there are 12 people with less than 16 years old and that have a salary, which does not make sense. At these ages the expected salary value was expected to be zero, which means that we were expecting the output of this code line to be zero.

# Check if people with less than 16 years old have a motor premium. This would not make sense as people with these ages do not have driving license.
dfOriginal['lobMotor'][dfOriginal['age']<16].count() # There are 12 people with less than 16 years old that have a motor premium, which does not make sense.

# Check if people with less than 18 years old have a household premium (we defined the age 18 years old as the minimum age for a person to get a house).
dfOriginal['lobHousehold'][dfOriginal['age']<18].count() # There are 116 people younger than 18 years old who have a household premium, which does not make sense.

# Check if people with less than 16 years old have a work compensation premium, which does not make sense. The minimum age to start working is 16 years old.
dfOriginal['lobWork'][dfOriginal['age']<16].count() # There are 12 people younger than 16 years old that have a work compensation premium.

# Final Decision: drop birthday and age columns - they do not make any sense when considering other variables in the data set.

# Create a column for year salary (useful to check the next incoherence).
# Create a column with the total premiums (useful to check the next incoherence)
# Check if the 30% of the year salary is higher than the lobTotal
# Check if the 50% of the year salary is higher than the lobTotal
dfOriginal['yearSalary']=dfOriginal['salary']*12
dfOriginal['lobTotal']=dfOriginal['lobMotor']+dfOriginal['lobHousehold']+dfOriginal['lobHealth']+dfOriginal['lobLife']+dfOriginal['lobWork']
dfOriginal[dfOriginal['yearSalary']*0.3<dfOriginal['lobTotal']] # There are 14 people that spend more than 30% of the year salary in the total of premiums.
dfOriginal['id'][dfOriginal['yearSalary']*0.5<dfOriginal['lobTotal']].count() # There are 2 people that spend more than 50% of the year salary in the total of premiums, which migh be considered strange. It is not normal for a person spending more than 50% of the salary in premiums.

# Check if the year salary is higher than the lobTotal. If not, it does not make sense.
dfOriginal['id'][dfOriginal['yearSalary']<dfOriginal['lobTotal']].count() # There is one person that has a salary lower than the lobTotal, which does not make sense.
# We decided to add this customer to an incoherence column.
dfOriginal['incoherences']=np.where(dfOriginal['yearSalary']<dfOriginal['lobTotal'],1,0)
dfOriginal['Others']=np.where(dfOriginal['yearSalary']<dfOriginal['lobTotal'],1,dfOriginal['Others'])

#---------------CREATE A DATA FRAME TO WORK ON (WITH NO INCOHERENCES AND NO OUTLIERS)------------------------
# Take outliers, strange values and incoherences out.
# Drop columns that are not needed: because have all values zero and other reasons
dfWork=dfOriginal[dfOriginal['Others']==0]
dfWork=dfWork.drop(columns=['age', 'birthday','incoherences','Strange_birthday','Others','strange_firstPolicy','Outliers_salary','df_OutliersLow_cmv','df_OutliersHigh_cmv','Outliers_lobMot','Outliers_lobHousehold','Outliers_lobHealth','Outliers_lobWork'])

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
dfWork[dfWork['lobHealth'].isnull()]
dfWork['lobHealth'] = np.where(dfWork['lobHealth'].isnull(),0,dfWork['lobHealth'])
dfWork[dfWork['lobLife'].isnull()]
dfWork['lobLife'] = np.where(dfWork['lobLife'].isnull(),0,dfWork['lobLife'])
dfWork[dfWork['lobWork'].isnull()]
dfWork['lobWork'] = np.where(dfWork['lobWork'].isnull(),0,dfWork['lobWork'])

# Check again Nan values by row:
dfNan = dfWork.drop(columns = ['yearSalary', 'lobTotal'])
dfNan['Nan'] = dfNan.isnull().sum(axis=1)
dfNan['Nan'].value_counts()

# Recalculate the column lobTotal (as there are no Null values on the lob variables anymore)
dfWork['lobTotal']=dfWork['lobMotor']+dfWork['lobHousehold']+dfWork['lobHealth']+dfWork['lobLife']+dfWork['lobWork']
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
# The variable salary explains the variable children.
# The variable lobMotor explains the variable children.
# Lets use the variables lobmotor and salary to explain the variable children and to treat the Nan values through the KNN:

# dfChildren: to treat Children Nan values
dfChildren=dfWork[['id','salary','lobMotor','lobHealth','children']]
dfChildren['children'][dfChildren['salary'].isna()].isna().sum()
dfChildren['children'][dfChildren['lobMotor'].isna()].isna().sum()
# There is no individual that has both salary and children null.
# There is no individual that has both LobMotor and children null (this would never happen as we have already treated null values for lob variables)

# Delete rows that have salary and/or lobMotor null.
dfChildren = dfChildren[~((dfChildren['salary'].isna())|(dfChildren['lobMotor'].isna()))]

# Apply the KNN Function:
dfWork=KNClassifier(dfWork=dfWork,myDf=dfChildren, treatVariable='children', expVariables=['salary','lobMotor','lobHealth'],K=5,weights='distance', metric='minkowski',p=1)
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
dfWork['binEducation']=np.where(((dfWork['binEducation']=='1 - Basic')|(dfWork['binEducation']=='2 - High School')),0,1)

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
imputer = KNNImputer(n_neighbors=5)
dfSalary=pd.DataFrame(imputer.fit_transform(dfSalary), columns=dfSalary.columns)
# Check again nan values: 
dfSalary.isna().sum()
# Replace column in the original data frame:
dfSalary=dfSalary[['id','salary']]
dfSalary=dfSalary.rename(columns={'salary':'salary_x'})
dfWork= reduce(lambda left,right: pd.merge(left, right, on='id', how='left'), [dfWork,dfSalary])
dfWork['salary']=np.where(dfWork['salary'].isna(),dfWork['salary_x'],dfWork['salary'])
dfWork=dfWork.drop(columns=['salary_x'])

#Check again Null values.
dfWork.isna().sum()
# As expected there are still two null values on salary. These are the one that we excluded because salary and firstPolicy were both null.

# Lets treat these two observations. For this we cannot use firstPolicy as an explainable variable. Lets just use the Lob variables:
dfSalary=dfWork[['id','salary','lobMotor','lobHousehold','lobHealth','lobLife','lobWork']]
dfSalary=pd.DataFrame(imputer.fit_transform(dfSalary), columns=dfSalary.columns)
# Check again nan values: 
dfSalary.isna().sum()
# Replace column in the original data frame:
dfSalary=dfSalary[['id','salary']]
dfSalary=dfSalary.rename(columns={'salary':'salary_x'})
dfWork= reduce(lambda left,right: pd.merge(left, right, on='id', how='left'), [dfWork,dfSalary])
dfWork['salary']=np.where(dfWork['salary'].isna(),dfWork['salary_x'],dfWork['salary'])
dfWork=dfWork.drop(columns=['salary_x'])
#Check again Null values.
#Recalculate column yearSalary.
#Check again Null values.
dfWork.isna().sum() # zero null values on salary as expected
dfWork.isna().sum() # no null values on salary
dfWork['yearSalary']=dfWork['salary']*12
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
dfFirstPolicy=pd.DataFrame(imputer.fit_transform(dfFirstPolicy), columns=dfFirstPolicy.columns)

# Check again nan values: 
dfFirstPolicy.isna().sum()

# Replace column in the original data frame:
dfFirstPolicy=dfFirstPolicy[['id','firstPolicy']]
dfFirstPolicy=dfFirstPolicy.rename(columns={'firstPolicy':'firstPolicy_x'})
dfWork = reduce(lambda left,right: pd.merge(left, right, on='id', how='left'), [dfWork,dfFirstPolicy])
dfWork['firstPolicy']=np.where(dfWork['firstPolicy'].isna(),dfWork['firstPolicy_x'],dfWork['firstPolicy'])
dfWork = dfWork.drop(columns=['firstPolicy_x'])
dfWork.isna().sum()

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
dfWork['motorRatioLOB']=dfWork["lobMotor"]/dfWork['lobTotal']
dfWork['householdRatioLOB']=dfWork["lobHousehold"]/dfWork['lobTotal']
dfWork['healthRatioLOB']=dfWork["lobHealth"]/dfWork['lobTotal']
dfWork['lifeRatioLOB']=dfWork["lobLife"]/dfWork['lobTotal']
dfWork['workCRatioLOB']=dfWork["lobWork"]/dfWork['lobTotal']

# lobTotal/salary
dfWork['ratioSalaryLOB']=dfWork['lobTotal']/dfWork['salary']

# Years has been a customer= 1998-firstPolicy compare with 2016
dfWork['YearsWus1998']=1998-dfWork['firstPolicy']
dfWork['YearsWus2016']=2016-dfWork['firstPolicy']

dfOriginal['CMV_Mean_corrected']=(dfOriginal['cmv']+25)
##### create categorical values of cmv corrected
dfWork['catCMV_Mean_corrected']=np.where(dfWork['cmv']<-25,'losses',None)
dfWork['catCMV_Mean_corrected']=np.where(dfWork['cmv']>-25,'profitable',dfWork['catCMV_Mean_corrected'])
dfWork['catCMV_Mean_corrected']=np.where(dfWork['cmv']==-25,'neutrals',dfWork['catCMV_Mean_corrected'])
#check cancel o see if categorical cancel is a thing
df=pd.DataFrame(dfWork[["CancelTotal","lobMotor","lobHousehold","lobHealth","lobLife","lobWork"]][dfWork['CancelTotal'] > 0 ])
TODO: canceled  0,1,2,...
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

#TODO: levar para a frente e multiplicar pelos anos... parece não fazer muito sentido...
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
df=pd.DataFrame(dfOriginal['catCMV_Mean_corrected'].value_counts().sort_index())
df.reset_index(level=0, inplace=True)
df=df.rename(columns={'index':'catCMV_Mean_corrected','catCMV_Mean_corrected':'Individuals'})


data= go.Bar(x=df['catCMV_Mean_corrected'],y=df['Individuals'])
layout = go.Layout(title='Customer Monetary Value Mean Corrected Variable',template='simple_white',
        xaxis=dict(title='CMV Value',showgrid=True),yaxis=dict(title='Number of Individuals',showgrid=True))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)

####################################################################################################################
#Compare Variables


#---------------------------------------------- MULTIDIMENSIONAL OUTLIERS -------------------------------------------------#

dfMultiOut=dfWork[['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork']]
# Min max: outliers already treated and for variables to be at the same scale.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
multiNorm = scaler.fit_transform(dfMultiOut)
multiNorm = pd.DataFrame(multiNorm, columns = dfMultiOut.columns)

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
labelsKmeans.columns =  ['LabelsKmeans']
labelsKmeans

outClientsCluster = pd.DataFrame(pd.concat([multiNorm, labelsKmeans],axis=1), 
                        columns=['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork','LabelsKmeans'])

# Hierarchical clustering:
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(multiNorm,
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
           show_leaf_counts=True,color_threshold=50, above_threshold_color='k',no_labels =True)

hClustering = AgglomerativeClustering(n_clusters = 30,
                                      affinity = 'euclidean',
                                      linkage = 'ward')

# With 30 clusters the normal behavior would be to have 300 observations in each cluster.
multiHC = hClustering.fit(multiClusters)

# clusters hcclusters aos quais cada k means clusters pertence?
# Quero aceder às labels do k means ao mesmo tempo que acedo às labels do hc - coluna 1: kmeans; coluna 2: hc

labelsHC = pd.DataFrame(multiHC.labels_)
labelsHC.columns =  ['LabelsHC']
labelsHC.reset_index(level=0, inplace=True)
labelsHC=labelsHC.rename(columns={'index':'LabelsKmeans'})
labelsHC['LabelsKmeans']=labelsHC['LabelsKmeans']+1

outClientsCluster=outClientsCluster.merge(labelsHC, left_on='LabelsKmeans', right_on='LabelsKmeans')
outClientsCluster=outClientsCluster['LabelsHC'].value_counts().sort_values()

# Pair plot to check ni dimensional outliers:
plt.figure()
sb.pairplot(dfWork, vars=['firstPolicy','salary', 'cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
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
        sb.violinplot(x='labelsKmeans', y=i, data=dfKmeans, scale='width', inner='quartile')
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
    lista=["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]
    fig=plt.figure()
    fig.suptitle('Box Plots by Variable and Cluster (K-means + Hierarchical)')
    for i in lista:
        plt.subplot2grid((1,len(lista)),(0,lista.index(i)))
        sb.violinplot(x='labelsKmeansHC2', y=i, data=dfKmeansHC, scale='width', inner='quartile')
        plt.xlabel('Cluster Number')
        plt.title(str(i))
        plt.ylabel('')
    plt.tight_layout()
    plt.plot()
    
    # Silhouette Graph:   
    silhouette_avg = silhouette_score(dfNorm,dfKmeansHC['labelsKmeansHC2']) 
    print("For number of cluster (k) =", nHC,
              "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dfNorm,dfKmeansHC['labelsKmeansHC2'])
    cluster_labels = dfKmeansHC['labelsKmeansHC2']
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
        sb.violinplot(x='labelsSomHC2', y=i, data=dfSomHC, scale='width', inner='quartile')
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
                                  n_init=10,
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
        sb.violinplot(x='labelsEm', y=i, data=dfEM, scale='width', inner='quartile')
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
                                      quantile=0.2,
                                      n_samples=1000)
    
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
        sb.violinplot(x='labelsMs', y=i, data=dfMeanShift, scale='width', inner='quartile')
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
    db = DBSCAN(eps= 1, #radius (euclidean distance)
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
        sb.violinplot(x='labelsDB', y=i, data=dfDB, scale='width', inner='quartile')
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
    lista=['education','binEducation','children','catClaims']
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
dfEngage = dfWork[[  'firstPolicy',
                      'salary',
                      'cmv',
                      'yearCustomer',
                      #'claims' # use claims only when ploting for comparing clusters
                      ]] 

dfEngageCat = dfWork[['education',
                         'binEducation',
                         'children',
                         'catClaims',
                         #'catCMV' #TODO: Implement
                         ]]

dfAffinity= dfWork[['lobMotor',
                  'lobHousehold',
                  'lobHealth',
                  'lobLife',
                  'lobWork']]

dfAffinityRatio=dfWork[['motorRatio',
                      'householdRatio',
                      'healthRatio',
                      'lifeRatio',
                      'workCRatio']]

########################################### K-means ##########################################
# Normalization: min-max
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
dfEngageKmeans=dfEngage
engageNorm = scaler.fit_transform(dfEngageKmeans)
engageNorm = pd.DataFrame(engageNorm, columns = dfEngageKmeans.columns)
engageNorm = engageNorm.rename(columns={"firstPolicy": "firstPolictStd", "salary": "salaryStd", "cmv": "cmvStd", "yearCustomer":"yearCustomerStd"}, errors="raise")
dfEngageKmeans=pd.DataFrame(pd.concat([dfEngageKmeans, engageNorm],axis=1), 
                        columns=["firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
dfEngageKmeans=pd.DataFrame(pd.concat([dfWork['id'], dfEngageKmeans],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])

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
plt.xlabel("Clusters")
plt.ylabel("Within- Cluster Sum of Squares")
plt.title('Elbow Graph')
plt.plot(dfClusters.Num_clusters,dfClusters.Cluster_errors,marker='o') # There is evidence that we should keep 2 clusters.

# We also tried to make 3 clusters instead to check if with 3 clusters we could differentiate according to salary and cmv.
dfEngageKmeans=kmeans_funct(dfKmeans=dfEngageKmeans,dfNorm=engageNorm, n=2,returndf=True)
# The 3 clusters continue not differentiating salary and cmv.
# Final Decision: keep 2 clusters when applying k-means cluster technique.

########################################### K-means + Hierarchical ##########################################
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

dfEngageKmeansHC=dfEngage
dfEngageKmeansHC=pd.DataFrame(pd.concat([dfEngageKmeansHC, engageNorm],axis=1), 
                        columns=["firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
dfEngageKmeansHC=pd.DataFrame(pd.concat([dfWork['id'], dfEngageKmeansHC],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
dfEngageKmeansHC=kmeansHC_funct(dfNorm=engageNorm,dfKmeansHC=dfEngageKmeansHC,nkmeans=int(round(math.sqrt(10261),0)),nHC=3,returndf=True)

########################################### SOM + Hierarchical ##########################################
from sompy.sompy import SOMFactory

dfEngageSomHC=dfEngage
dfEngageSomHC=pd.DataFrame(pd.concat([dfEngageSomHC, engageNorm],axis=1), 
                        columns=["firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
dfEngageSomHC=pd.DataFrame(pd.concat([dfWork['id'], dfEngageSomHC],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
names = ["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]

dfEngageSomHC=SomHC_funct(dfNorm=engageNorm,dfSomHC=dfEngageSomHC,names=names,nHC=3,returndf=True)

########################################### EM ##########################################
from sklearn import mixture
dfEngageEM=dfEngage
dfEngageEM=pd.DataFrame(pd.concat([dfEngageEM, engageNorm],axis=1), 
                        columns=["firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
dfEngageEM=pd.DataFrame(pd.concat([dfWork['id'], dfEngageEM],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])

dfEngageEM=EM_funct(dfNorm=engageNorm, dfEM=dfEngageEM, n=3,returndf=True)
EM_funct()
#Not used yet:
# Likelihood value
#EM_score_samp = pd.DataFrame(gmm.score_samples(engageNorm))
# Probabilities of belonging to each cluster.
#EM_pred_prob = pd.DataFrame(gmm.predict_proba(engageNorm))

########################################### Mean Shift ##########################################
from sklearn.cluster import MeanShift, estimate_bandwidth
dfEngageMs=dfEngage
dfEngageMs=pd.DataFrame(pd.concat([dfEngageMs, engageNorm],axis=1), 
                        columns=["firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
dfEngageMs=pd.DataFrame(pd.concat([dfWork['id'], dfEngageMs],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])

dfEngageMs=MeanShift_funct(dfNorm=engageNorm,dfMeanShift=dfEngageMs,returndf=True)

########################################### DB- Scan ##########################################
# Esquecer isto porque não faz sentido. As densidades são muito similares.
from sklearn.cluster import DBSCAN
dfEngageDb=dfEngage
dfEngageDb=pd.DataFrame(pd.concat([dfEngageDb, engageNorm],axis=1), 
                        columns=["firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])
dfEngageDb=pd.DataFrame(pd.concat([dfWork['id'], dfEngageDb],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"])

dfEngageDb=DBScan_funct(dfNorm=engageNorm,dfDB=dfEngageDb,returndf=True)

########################################### K-modes ##########################################
from kmodes.kmodes import KModes
dfEngageCatKmodes=dfEngageCat
dfEngageCatKmodes=pd.DataFrame(pd.concat([dfWork['id'], dfEngageCatKmodes],axis=1), 
                        columns=['id','education','binEducation','children','catClaims'])
kmodesChange = dfEngageCatKmodes[['education','binEducation','children','catClaims']].astype('str')

dfEngageCatKmodes=Kmodes_funct(n=3,dfKmodesChange=kmodesChange,dfKmodes=dfEngageCatKmodes,returndf=True)








<<<<<<< HEAD
=======
# Violin Figure:
lista=["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]
fig=plt.figure()
fig.suptitle('Box Plots by Variable and Cluster')
for i in lista:
    plt.subplot2grid((1,4),(0,lista.index(i)))
    sb.violinplot(x='labelsKmeansEngageK3', y=i, data=dfEngageKmeansK3, scale='width', inner='quartile',)
plt.tight_layout()
plt.plot()
>>>>>>> 9d38440ed74ffcdc2cfd56f87084179a7ae158a1






















































































































Hclustering = AgglomerativeClustering(n_clusters = 30,
                                      affinity = 'euclidean',
                                      linkage = 'ward')

# With 30 clusters the normal behavior would be to have 300 observations in each cluster.
MultiHC = Hclustering.fit(multiClusters)

# clusters hcclusters aos quais cada k means clusters pertence?
# Quero aceder às labels do k means ao mesmo tempo que acedo às labels do hc - coluna 1: kmeans; coluna 2: hc

labelsHC = pd.DataFrame(MultiHC.labels_)
labelsHC.columns =  ['LabelsHC']
labelsHC.reset_index(level=0, inplace=True)
labelsHC=labelsHC.rename(columns={'index':'LabelsKmeans'})
labelsHC['LabelsKmeans']=labelsHC['LabelsKmeans']+1

OutClientsCluster=OutClientsCluster.merge(labelsHC, left_on='LabelsKmeans', right_on='LabelsKmeans')
OutClientsCluster=OutClientsCluster['LabelsHC'].value_counts().sort_values()










































    



















#ANOTHER WAY OF IMPUTATION KNN: CHILDREN:
from sklearn.impute import KNNImputer

#Check null values on children:
dfWork['children'].isna().sum()     #There are 21 Nan values

# Which variables better explain the variable children?
plt.figure()
sb.pairplot(dfWork2, vars=['firstPolicy','salary','cmv','claims','lobHousehold'], hue='children')
plt.show()
plt.figure()
sb.pairplot(dfWork2, vars=['lobMotor','lobHealth','lobLife','lobWork','lobTotal','yearSalary'], hue='children')
plt.show()
# The variable salary explains the variable children.
# The variable lobMotor explains the variable children.
# Lets use the variables lobmotor and salary to explain the variable children and to treat the Nan values through the KNN:

# Create a data frame that does not have null values on salary neither on lobMotor
dfWork2=dfWork[~((dfWork['salary'].isna()) | (dfWork['lobMotor'].isna()))]
#keep only 3 columns
children_KNN=dfWork2[['id','salary','lobMotor','children']]
#train the model and impute
imputer =KNNImputer(n_neighbors=2)
children_KNN2=imputer.fit_transform(children_KNN)


dfCorr=pd.DataFrame(dfWork,columns=['firstPolicy','salary','cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
corrMatrix=dfCorr.corr(method ='pearson')

plt.figure()
fig, ax = plt.subplots(figsize=(10,10))
sb.heatmap(corrMatrix,annot=True,fmt=".3f")
plt.show()
# As this only gives information about the linear correlation, we are going to check in more detail with the pairplot
plt.figure()
dfCorr2=dfCorr.dropna()
sb.pairplot(x='',diag_kind='kde',kind='scatter',palette='husl', hue='livingArea', data=dfWork)
plt.show()


















#Check if there is an lob=0 for first policy Nan. It would make sense if that happened, because that would be the case in which a person does not have a first policy and, therefore, would not have lob value (lob=0).
dfWork['lobTotal'][dfWork['firstPolicy'].isnull()].value_counts()
# There is no such value. Therefore, the null values in first policy cannot be treated based on this.

#KNN Treatment of Nan
# Correlation matrix: to check if there linear correlations between variables
dfCorr=pd.DataFrame(dfWork,columns=['firstPolicy','salary','cmv','claims','lobMotor','lobHousehold','lobHealth','lobLife','lobWork'])
dfCorr=dfCorr.corr(method ='pearson')
mask = np.zeros_like(dfCorr)
mask[np.triu_indices_from(mask)] = True
with sb.axes_style("white"):
    fig, ax = plt.subplots(figsize=(10,10))
    ax = sb.heatmap(dfCorr, annot=True, mask=mask, fmt="0.3", square=True,  cbar_kws={'label': 'Colorbar'})






sb.heatmap(dfCorr,annot=True,fmt=".3f", cbar_kws={'label': 'Colorbar'})
plt.show()




# As this only gives information about the linear correlation, we are going to check in more detail with the pairplot
plt.figure()
dfCorr2=dfCorr.dropna()
sb.pairplot(dfCorr2,diag_kind='kde',kind='scatter',palette='husl')
plt.show()

# Through the created pairplot we can take very useful information about the correlation between variables:
# Check in more detail specific correlations: 

# 




















#Line chart
from matplotlib import pyplot as plt
import seaborn as sb
d = {'count': countList, 'ages': agesList}
dfIdades = pd.DataFrame(data=d)
dfIdades
g=sb.relplot(x='ages',y='count',data=dfIdades)
g.fig.autofmt_xdate()

dfOriginal=dfOriginal.drop(['idade'],axis=1)
dfOriginal['age']=2016-dfOriginal['birthday']
dfOriginal['age'].value_counts().plot()

dfOriginal['age'].hist(bins=30)

dfOriginal['age'].describe()




#Check null values
dfOriginal.isnull().sum()

#Check categories in livingArea variable
dfOriginal["education"].unique()
#-----------------DATA TYPES------------------#
dfOriginal.dtypes
dfOriginal.loc[:,"firstPolicy"]=pd.to_numeric(dfOriginal.loc[:,"firstPolicy"],errors='coerce')
dfOriginal.loc[3,"firstPolicy"].dtype
dfOriginal.iloc[] = dfOriginal.iloc[:,2].astype('int64')
dfOriginal.iloc[:,5]=dfOriginal.iloc[:,5].astype('int64')
dfOriginal.iloc[:,6]=dfOriginal.iloc[:,6].astype('int64')

#------------------UNDERSTAND VARIABLES-------------------#

#Pivot tables
#Salary mean having in mind children and livingArea 
pd.pivot_table(dfOriginal,
               values='salary',
               index='livingArea',
               columns=['children'],
               aggfunc='mean',
               margins=True)
#number of individuals having in mind education and livingArea 
pd.pivot_table(dfOriginal,
               values='salary',
               index='livingArea',
               columns=['education'],
               aggfunc='count',margins=True) #ver em proporção

#Variables univariate Distributions:
sb.distplot(dfOriginal['claims'],kde=False)
plt.show()

#Plots:
from matplotlib import pyplot as plt
import seaborn as sb

#Check correlations between variables: Pairgrid

g=sb.PairGrid(df1)
g.map_diag(plt.hist) #diagonal:histogramas
g.map_offdiag(plt.scatter);#fora da diagonal scatterplots
plt.show()












#-----------------OUTLIERS------------------#
#Check distribution plots of variables (to see presence of outliers)
import seaborn as sb
sb.boxplot(x=dfOriginal["birthday"]) 
sb.boxplot(x=dfOriginal["education"]) 
sb.boxplot(x=dfOriginal["salary"]) 
sb.boxplot(x=dfOriginal["livingArea"]) 
sb.boxplot(x=dfOriginal["children"]) 
sb.boxplot(x=dfOriginal["cmv"]) 
sb.boxplot(x=dfOriginal["claims"]) 
sb.boxplot(x=dfOriginal["lobMotor"]) 
sb.boxplot(x=dfOriginal["lobHousehold"]) 
sb.boxplot(x=dfOriginal["lobHealth"]) 
sb.boxplot(x=dfOriginal["lobLife"])
sb.boxplot(x=dfOriginal["lobWork"]) 

#Create a dataframe with outliers
q1=dfOriginal["salary"].quantile(0.25)
q3=dfOriginal["salary"].quantile(0.75)
iqr=q3-q1 #Interquartile range
fence_low=q1-1.5*iqr
fence_high=q3+1.5*iqr
df_out=dfOriginal.loc[(dfOriginal["salary"] < fence_low) | (dfOriginal["salary"] > fence_high)]
dfOriginal=dfOriginal.loc[(dfOriginal["salary"] > fence_low) & (dfOriginal["salary"] < fence_high)]

def remove_outliers(df,df_out,column_name):  
    q1=df[column_name].quantile(0.25)
    q3=df[column_name].quantile(0.75)
    iqr=q3-q1 #Interquartile range
    fence_low=q1-1.5*iqr
    fence_high=q3+1.5*iqr
    df_out=df_out.append(df.loc[(df[column_name] < fence_low) | (df[column_name] > fence_high)])
    df=df.loc[(df[column_name] > fence_low) & (df[column_name] < fence_high)]
    return df_out,df


df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"birthday")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"education")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"livingArea")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"children")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"cmv")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"claims")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"lobMotor")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"lobHousehold")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"lobHealth")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"lobLife")
df_out,dfOriginal=remove_outliers(dfOriginal,df_out,"lobWork")

dfOriginal.shape[0]
df_out.shape[0] #1273 outliers in the df_out
#-----------------TREAT NAN VALUES------------------#
    #variable: firstPolicy
    for i, index in dfOriginal.iterrows():
            if str(dfOriginal.iloc[i,1])=='nan':
                dfOriginal.iloc[i,1]=round(dfOriginal.iloc[:,1].mean(),0)
    
    #variable: birthday
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,2])=='nan':
            dfOriginal.iloc[i,2]=round(dfOriginal.iloc[:,2].mean(),0)
    
    #variable: education - do the proportion and put the values
    my_count=dfOriginal.groupby(by=["education"])["id"].count()#check class with more individuals
    my_count
    nrows=my_count[1]+my_count[2]+my_count[3]+my_count[0]
    pbasic=round(my_count[0]/nrows*dfOriginal.isnull().sum()["education"],0)
    pHS=round(my_count[1]/nrows*dfOriginal.isnull().sum()["education"],0)
    pBSc_MSc=round(my_count[2]/nrows*dfOriginal.isnull().sum()["education"],0)
    pPhd=round(my_count[3]/nrows*dfOriginal.isnull().sum()["education"],0)
    
    n=0
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,3])=='nan':
            if n<2: 
                dfOriginal.iloc[i,3]='1 - Basic'
                n=n+1
            elif n>=2 and n<8:
                dfOriginal.iloc[i,3]='2 - High School'
                n=n+1
            elif n>=8 and n<16:
                dfOriginal.iloc[i,3]='3 - BSc/MSc'
                n=n+1
            else:
                dfOriginal.iloc[i,3]='4 - PhD'
                n=n+1
        
    #variable: Salary: put mean
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,4])=='nan':
            dfOriginal.iloc[i,4]=round(dfOriginal.iloc[:,4].mean(),0)
    
    #variable: Geographical area: see the most lived area:
    my_countGA=dfOriginal.groupby(by=["livingArea"])["id"].count()#check class with more individuals
    max_GA=my_countGA.max()
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,5])=='nan': dfOriginal.iloc[i,5]=max_GA
    
    #variable: children: calculate proportion?
    my_count_child=dfOriginal.groupby(by=["children"])["id"].count()
    nrows_child=my_count_child[1]+my_count_child[0]
    pchild0=round(my_count_child[0]/nrows_child*dfOriginal.isnull().sum()["children"],0)
    pchild1=round(my_count_child[1]/nrows_child*dfOriginal.isnull().sum()["children"],0)
    pchild0
    pchild1
    n=0
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,6])=='nan':
            if n<6: 
                dfOriginal.iloc[i,6]=0
                n=n+1
            else:
                dfOriginal.iloc[i,6]=1
                n=n+1
    
    #variable: lobMotor, lobHealth, lobLife, lobWork: valor médio para que não tenham um grande impacto na análise
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,9])=='nan':
            dfOriginal.iloc[i,9]=round(dfOriginal.iloc[:,9].mean(),0)
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,11])=='nan':
            dfOriginal.iloc[i,11]=round(dfOriginal.iloc[:,11].mean(),0)
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,12])=='nan':
            dfOriginal.iloc[i,12]=round(dfOriginal.iloc[:,12].mean(),0)
    for i, index in dfOriginal.iterrows():
        if str(dfOriginal.iloc[i,13])=='nan':
            dfOriginal.iloc[i,13]=round(dfOriginal.iloc[:,13].mean(),0)












dfOriginal.shape[0]
#Create a data frame for the incoherent values
df_inc=dfOriginal.loc[dfOriginal["birthday"]>dfOriginal["firstPolicy"]]
df_inc.shape[0]
dfOriginal=dfOriginal.loc[dfOriginal["birthday"]<=dfOriginal["firstPolicy"]]


#Check incoherences in lobMotor variable
    countmotor=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,"lobMotor"]<0: countmotor+=1
    countmotor #No incoherences

#Check incoherences in lobHousehold variable:
    counthousehold=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,"lobHousehold"]<0: counthousehold+=1
    counthousehold #there are 1007 observations with incoherences
    #Add to df_inc and delete from dfOriginal
    df_inc=df_inc.append(dfOriginal.loc[dfOriginal["lobHousehold"]<0])
    dfOriginal=dfOriginal.loc[dfOriginal["lobHousehold"]>=0]

#Check incoherences in lobHealth variable:
    counthealth=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,"lobHealth"]<0: counthealth+=1
    counthealth #No incoherences

#Check incoherences in lobLife variable
    countlife=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,"lobLife"]<0: countlife+=1
    countlife #there are 462 observations with incoherences
    #Add to df_inc and delete from dfOriginal
    df_inc=df_inc.append(dfOriginal.loc[dfOriginal["lobLife"]<0])
    dfOriginal=dfOriginal.loc[dfOriginal["lobLife"]>=0]

#Check incoherences in lobWork variable
    countWork=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,'Premiums in LOB: Work Compensations']<0: countWork+=1
    countWork #there are 550 observations with incoherences
    #Add to df_inc and delete from dfOriginal
    df_inc=df_inc.append(dfOriginal.loc[dfOriginal['Premiums in LOB: Work Compensations']<0])
    dfOriginal=dfOriginal.loc[dfOriginal['Premiums in LOB: Work Compensations']>=0]

#Check if column children is all 0's and 1's
countchild=0
for i, index in dfOriginal.iterrows():
    if (dfOriginal.loc[i,'Has Children (Y=1)']!=0 and dfOriginal.loc[i,"children"]!=1): countchild+=1
countchild

#Check if all salaries are positive
salary=0
for i, index in dfOriginal.iterrows():
    if dfOriginal.loc[i,"salary"]<0: salary+=1
salary

#Check if claims variable is >0
    claims=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,"claims"]<0: claims+=1
    claims #No incoherences



#-----------------NEW VARIABLES------------------#

#Variable1: Sum of all LOB:
dfOriginal['SumLOB']=dfOriginal["lobMotor"]+dfOriginal["lobHousehold"]+dfOriginal["lobHealth"]+dfOriginal["lobLife"]+dfOriginal["lobWork"]
#Outra forma: dfOriginal['SumLOB'] = dfOriginal.apply(lambda row: row["lobMotor"] + row["lobHousehold"]+row["lobHealth"]+row["lobLife"]+row["lobWork"], axis=1)

#Variable2: ratios of LOB:
dfOriginal['motorRatio']=dfOriginal["lobMotor"]/dfOriginal['SumLOB']
dfOriginal['householdRatio']=dfOriginal["lobHousehold"]/dfOriginal['SumLOB']
dfOriginal['healthRatio']=dfOriginal["lobHealth"]/dfOriginal['SumLOB']
dfOriginal['lifeRatio']=dfOriginal["lobLife"]/dfOriginal['SumLOB']
dfOriginal['workCRatio']=dfOriginal["lobWork"]/dfOriginal['SumLOB']



        
































#Scatter plots to see correlation between variables:
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(dfOriginal["salary"])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()






# Define seed
# Apply k-means
my_seed=100
kmeans = KMeans(n_clusters=2, # tried here with 3 clusters
                random_state=0,
                n_init = 20,
                max_iter = 300,
                init='k-means++').fit(engageNorm)

# Check the Clusters (Centroids).
kmeansClustersEngage=kmeans.cluster_centers_
kmeansClustersEngage

labelsKmeansEngage = pd.DataFrame(kmeans.labels_)
labelsKmeansEngage.columns =  ['labelsKmeansEngage']
labelsKmeansEngage

dfEngageKmeans = pd.DataFrame(pd.concat([dfEngageKmeans, labelsKmeansEngage],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd",'labelsKmeansEngage'])

####### Visualization of results:

# Box Plots:
lista=["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]
fig=plt.figure()
fig.suptitle('Box Plots by Variable and Cluster (K-means)')
for i in lista:
    plt.subplot2grid((1,4),(0,lista.index(i)))
    sb.boxplot(x='labelsKmeansEngage', y=i, data=dfEngageKmeans)
    plt.xlabel('Cluster Number')
    plt.title(str(i))
    plt.ylabel('')
plt.tight_layout()
plt.plot()
# Through this visualization we could conclude that the 2 clusters differentiate a lot on the variables firstPolicy and yearCustomer. 
# However, these 2 clusters do not differentiate in the variables salary and cmv.

# Violin Plots:
lista=["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]
fig=plt.figure()
fig.suptitle('Violin Plots by Variable and Cluster (K-means')
for i in lista:
    plt.subplot2grid((1,4),(0,lista.index(i)))
    sb.violinplot(x='labelsKmeansEngage', y=i, data=dfEngageKmeans, scale='width', inner='quartile')
    plt.xlabel('Cluster Number')
    plt.title(str(i))
    plt.ylabel('')
plt.tight_layout()
plt.plot()

# Joy Plots:
# Resulta mais ou menos:
plt.figure(figsize=(16,10), dpi= 80)
fig, axes = joypy.joyplot(dfEngageKmeans, column=lista, by='labelsKmeansEngage', figsize=(14,10))
plt.plot()


# Apply k-means with the k as the square root of the obs number (k=101).
K=int(round(math.sqrt(10261),0))
my_seed=100
kmeans = KMeans(n_clusters=K, 
                random_state=0,
                n_init = 20, 
                max_iter = 300,
                init='k-means++').fit(engageNorm)

# Check the Clusters (Centroids).
kmeansHCCentroidsEngage=kmeans.cluster_centers_
kmeansHCCentroidsEngage

labelsKmeansHCEngage = pd.DataFrame(kmeans.labels_)
labelsKmeansHCEngage.columns =  ['labelsKmeansHCEngage']
labelsKmeansHCEngage

dfEngageKmeansHC = pd.DataFrame(pd.concat([dfEngageKmeansHC, labelsKmeansHCEngage],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd",'labelsKmeansHCEngage'])

# Create a dendogram to check how many of the 101 clusters should be retained:
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
hClustering = AgglomerativeClustering(n_clusters = 3, #try also with 3 and 4
                                      affinity = 'euclidean',
                                      linkage = 'ward').fit(kmeansHCCentroidsEngage)

labelsKmeansHCEngage2 = pd.DataFrame(hClustering.labels_)
labelsKmeansHCEngage2.columns =  ['labelsKmeansHCEngage2']
labelsKmeansHCEngage2.reset_index(level=0, inplace=True)
labelsKmeansHCEngage2=labelsKmeansHCEngage2.rename(columns={'index':'labelsKmeansHCEngage'})
labelsKmeansHCEngage2

# Join the new clusters to the data frame dfEngageKmeansHC with merge through the 'LabelsKmeansHCEngage'
dfEngageKmeansHC=dfEngageKmeansHC.merge(labelsKmeansHCEngage2, left_on='labelsKmeansHCEngage', right_on='labelsKmeansHCEngage')

####### Visualization of results:

# Box Plots:
lista=["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]
fig=plt.figure()
fig.suptitle('Box Plots by Variable and Cluster (K-means + Hierarchical)')
for i in lista:
    plt.subplot2grid((1,4),(0,lista.index(i)))
    sb.boxplot(x='labelsKmeansHCEngage2', y=i, data=dfEngageKmeansHC)
    plt.xlabel('Cluster Number')
    plt.title(str(i))
    plt.ylabel('')
plt.tight_layout()
plt.plot()
# Through this visualization we could conclude that the 2 clusters differentiate a lot on the variables firstPolicy and yearCustomer. 
# However, these 2 clusters do not differentiate in the variables salary and cmv.

# Violin Plots:
lista=["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]
fig=plt.figure()
fig.suptitle('Box Plots by Variable and Cluster (K-means + Hierarchical)')
for i in lista:
    plt.subplot2grid((1,4),(0,lista.index(i)))
    sb.violinplot(x='labelsKmeansHCEngage2', y=i, data=dfEngageKmeansHC, scale='width', inner='quartile')
    plt.xlabel('Cluster Number')
    plt.title(str(i))
    plt.ylabel('')
plt.tight_layout()
plt.plot()



# Apply SOM : 101 clusters 
names = ["firstPolictStd", "salaryStd","cmvStd","yearCustomerStd"]
sm = SOMFactory().build(data = engageNorm,
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

labelsSomHCEngage = pd.DataFrame(sm._bmu[0]) #101 clusters formed
labelsSomHCEngage.columns = ['labelsSomHCEngage']
labelsSomHCEngage

dfEngageSomHC = pd.DataFrame(pd.concat([dfEngageSomHC, labelsSomHCEngage],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd",'labelsSomHCEngage'])

# Get centroids:
somHCCentroidsEngage=sm.codebook.matrix

# Create a dendogram to check how many of the 101 clusters should be retained:
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
hClustering = AgglomerativeClustering(n_clusters = 3, # Try also with 4
                                      affinity = 'euclidean',
                                      linkage = 'ward').fit(somHCCentroidsEngage)

labelsSomHCEngage2 = pd.DataFrame(hClustering.labels_)
labelsSomHCEngage2.columns =  ['labelsSomHCEngage2']
labelsSomHCEngage2.reset_index(level=0, inplace=True)
labelsSomHCEngage2=labelsSomHCEngage2.rename(columns={'index':'labelsSomHCEngage'})
labelsSomHCEngage2

# Join the new clusters to the data frame dfEngageKmeansHC with merge through the 'LabelsKmeansHCEngage'
dfEngageSomHC=dfEngageSomHC.merge(labelsSomHCEngage2, left_on='labelsSomHCEngage', right_on='labelsSomHCEngage')




# n_components= <the number of elements you found before- centroids>
gmm = mixture.GaussianMixture(n_components = 3, # Evaluate this
                              init_params='kmeans', # {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
                              max_iter=1000,
                              n_init=10,
                              verbose = 1).fit(engageNorm)

labelsEmEngage = pd.DataFrame(gmm.predict(engageNorm))
labelsEmEngage.columns = ['labelsEmEngage']

dfEngageEM = pd.DataFrame(pd.concat([dfEngageEM, labelsEmEngage],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd",'labelsEmEngage'])







my_bandwidth = estimate_bandwidth(engageNorm,
                                  quantile=0.2,
                                  n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth,
               bin_seeding=True).fit(engageNorm)

labelsMsEngage = pd.DataFrame(ms.labels_) # 7 clusters created...
labelsMsEngage.columns = ['labelsMsEngage']



dfEngageMs = pd.DataFrame(pd.concat([dfEngageMs, labelsMsEngage],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd",'labelsMsEngage'])


db = DBSCAN(eps= 1, #radius (euclidean distance)
            min_samples=10).fit(engageNorm) # minimum number of points inside the radius.
labelsDbEngage = pd.DataFrame(db.labels_)
labelsDbEngage.columns = ['labelsDbEngage']
dfEngageDb = pd.DataFrame(pd.concat([dfEngageDb, labelsDbEngage],axis=1), 
                        columns=['id',"firstPolicy", "salary", "cmv", "yearCustomer","firstPolictStd", "salaryStd","cmvStd","yearCustomerStd",'labelsDbEngage'])



km = KModes(n_clusters=4, init='random', n_init=50, verbose=1) # Define the number of clusters wanted
clusters = km.fit_predict(kmodesChange)
labelsKmodesEngage=pd.DataFrame(km.labels_)
labelsKmodesEngage.columns = ['labelsKmodesEngage']

dfEngageCatKmodes = pd.DataFrame(pd.concat([dfEngageCatKmodes, labelsKmodesEngage],axis=1), 
                        columns=['id','education','binEducation','children','catClaims','labelsKmodesEngage'])

#Visualization:
lista=['education','binEducation','children','catClaims']
for i in lista:
    g = sb.catplot(i, col="labelsKmodesEngage", col_wrap=4,
                    data=dfEngageCatKmodes,
                    kind="count", height=3.5, aspect=0.8, 
                    palette='tab20')
plt.plot()

#Silhouette Graph:
# Applied to K-means

from sklearn.metrics import silhouette_samples 
from sklearn.metrics import silhouette_score 
import matplotlib.cm as cm

n_clusters=2
silhouette_avg = silhouette_score(engageNorm, kmeans.labels_) # kmeans.labels_ substitute
print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(engageNorm, kmeans.labels_)
cluster_labels = kmeans.labels_
y_lower = 100

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, engageNorm.shape[0] + (n_clusters + 1) * 10]) # substitute n_clusters!!

for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()    
    size_cluster_i=ith_cluster_silhouette_values. shape[0]
    y_upper = y_lower + size_cluster_i    
    color = cm.nipy_spectral(float(i) / n_clusters)
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





