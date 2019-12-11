# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:44:55 2019

@author: aSUS
"""

import pandas as pd
#!pip install modin[dask]
#import modin.pandas as pd # replaces pandas for parallel running, defaults to pandas when better method not available

import sqlite3
import numpy as np
from matplotlib import pyplot as plt
import math
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier

#""" my_path = 'C:/Users/aSUS/Documents/IMS/Master Data Science and Advanced Analytics with major in BA/Data mining/Projeto/insurance.db'
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

#Diretorias:
file='C:/Users/aSUS/Documents/IMS/Master Data Science and Advanced Analytics with major in BA/Data mining/Projeto/A2Z Insurance.csv'
#file= r'C:\Users\Pedro\Google Drive\IMS\1S-Master\Data Mining\Projecto\insurance.db'
#file='C:/Users/anaso/Desktop/Faculdade/Mestrado/Data Mining/Projeto/A2Z Insurance.csv'
#import csv file:

dfOriginal=pd.read_csv(file)
dfOriginal.head()

#Rename column
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
# 1. birthday:

# Plot birthday for a first visual analysis:
plt.figure()
dfOriginal['birthday'].value_counts().sort_index().plot() # there might be strange values on the birthday that are distorting the plot.
plt.show()

plt.figure()
dfOriginal['birthday'].hist() # The plot with the strange value is not perceptive
plt.show()

# Check variable values:
dfOriginal['birthday'].value_counts().sort_index() # strange value on the birthday: 1028

# Create a new column to indicate strange values as 1 and normal values as 0
# Explain the choice of 1900: (...)
dfOriginal['Strange_birthday'] = np.where(dfOriginal['birthday']<1900, 1,0)
dfOriginal['Others'] = np.where(dfOriginal['birthday']<1900, 1,0)
# Verify if the column was created as supposed
dfOriginal['Strange_birthday'].value_counts()

#Plot birthday variable with no strange values (where the new column equals zero):
plt.figure()
dfOriginal['birthday'][dfOriginal['Strange_birthday']==0].hist()
plt.show()

plt.figure()
dfOriginal['birthday'][dfOriginal['Strange_birthday']==0].value_counts().sort_index().plot(marker='o')
plt.show()

# 2. firstPolicy

# Plot firstPolicy for a first visual analysis:
plt.figure()
dfOriginal['firstPolicy'].value_counts().sort_index().plot() # there might be strange values on firstPolicy that are distorting the plot.
plt.show()

plt.figure()
dfOriginal['firstPolicy'].hist() # The plot with strange values is not perceptive
plt.show()

#Check variable values:
dfOriginal['firstPolicy'].value_counts().sort_index() # there is a strange value: 53784

# Create a new column to indicate strange values as 1 and normal values as 0:
# Explain the choice of 2016: (...)
dfOriginal['strange_firstPolicy']=np.where(dfOriginal['firstPolicy']>2016, 1,0)
dfOriginal['Others']=np.where(dfOriginal['firstPolicy']>2016, 1,dfOriginal['Others'])
# Verify if the column was created as supposed
dfOriginal['strange_firstPolicy'].value_counts()

#Plot firstPolicy variable with no strange values (where the created column equals zero):
plt.figure()
dfOriginal['firstPolicy'][dfOriginal['strange_firstPolicy']==0].hist()
plt.show()

plt.figure()
dfOriginal['firstPolicy'][dfOriginal['strange_firstPolicy']==0].value_counts().sort_index().plot(marker='o')
plt.show()

# 3. education (categorical variable)

# Create a variable to count the individuals per category:
counteducation=dfOriginal['education'].value_counts().sort_index()
counteducation

# Plot education variable:
plt.figure()
plt.bar(np.arange(len(counteducation.index)),counteducation)
plt.xticks(np.arange(len(counteducation.index)),counteducation.index)
plt.show()
# There is a considerable number of individuals with high education (BSc/MSc and PhD). 
# The number of individuals having PhD is not that high. We will consider later if it makes sense to join the categories BSc/MSc and PhD in a unique category.

# 4. salary
# To study this variable as it has different values that are not easily repeated through individuals, instead of counting by value as done with the previous cases, we decided to make the cumulative to be used for plotting.

# Plot salary for a first visual analysis:
plt.figure()
dfOriginal['salary'].value_counts().sort_index().plot() # there might be strange values on salary that are distorting the plot.
plt.show()

plt.figure()
dfOriginal['salary'].hist() # The plot with the strange value is not perceptive.
plt.show()

# Check variable values and create a variable for that:
countSalary = dfOriginal['salary'].value_counts().sort_index() # there are 2 out of the box values: 34490, 55215

# Create a new column to indicate outliers as 1 and normal values as 0:
# Explain chosen value for outliers (10000) (...)
dfOriginal['Outliers_salary']=np.where(dfOriginal['salary']>10000, 1,0)
dfOriginal['Others']=np.where(dfOriginal['salary']>10000, 1,dfOriginal['Others'])
# Verify if the column was created as supposed
dfOriginal['Outliers_salary'].value_counts()

# Create a variable with the cumulative salary values 
countSalaryCum = countSalary.cumsum()
countSalaryCum
#Plot the salary values and the cumulative values of salary
plt.figure()
countSalaryCum.plot()
plt.show()

# Plot salary non outliers values (where the created column equals zero):
plt.figure()
dfOriginal['salary'][dfOriginal['Outliers_salary']==0].hist()
plt.show()

plt.figure()
dfOriginal['salary'][dfOriginal['Outliers_salary']==0].value_counts().sort_index().plot(marker='o')
plt.show()

# Plot the cumulative salary non outliers values (where the created column equals zero):
plt.figure()
dfOriginal['salary'][dfOriginal['Outliers_salary']==0].value_counts().sort_index().cumsum().plot(marker='o')
plt.show()

# Check with log dist: as the usual behavior of a salary variable is a distribution with a heavy tail on the left side, usually it is applied a log transformation on the distribution in order to transform it to a normal distribution.
dfOriginal['logSalary'] = np.log(dfOriginal['salary'])
#count by log salary value
countLogSalary=dfOriginal['logSalary'].value_counts().sort_index()
#cumulative of log salary
countLogSalaryCum= countLogSalary.cumsum()
# Plot both log salary and cululative log salary
plt.figure()
countLogSalary.plot()
plt.show()

plt.figure()
countLogSalaryCum.plot()
plt.show()

# Log distributon: not applicable as original distribution is already normal (it does not follow the usual behavior).
# Drop created column as it will not be used
dfOriginal=dfOriginal.drop(['logSalary'], axis=1)

# 5. livingArea (categorical variable)

# Create a variable to count the individuals per category:
countlivingArea=dfOriginal['livingArea'].value_counts().sort_index()
countlivingArea

#Create a bar chart that shows the number of individuals per living area
plt.figure()
plt.bar(np.arange(len(countlivingArea.index)),countlivingArea)
plt.xticks(np.arange(len(countlivingArea.index)),countlivingArea.index)
plt.show()
# As we dont have any information on the location of each category of living area variable, we probably will not be able to suggest modifications such as joining categories.

# 6. children

# Create a variable to count the individuals per category:
countchildren=dfOriginal['children'].value_counts().sort_index()
countchildren
# Create a bar chart that shows the number of individuals with and without children
plt.figure()
plt.bar(np.arange(len(countchildren.index)),countchildren)
plt.xticks(np.arange(len(countchildren.index)),countchildren.index)
plt.show()
# There are more individuals with children that without.

# 7. cmv 
# To study this variable as it has different values that are not easily repeated through individuals, instead of counting by value, we decided to make the cumulative to plot, as done with the salary variable.

#Plot cmv for a first visual analysis:
plt.figure()
dfOriginal['cmv'].value_counts().sort_index().plot() # there might be strange values on cmv that are distorting the plot.
plt.show()

plt.figure()
dfOriginal['cmv'].hist() # The plot with the strange value is not perceptive
plt.show()

# Create a variable that counts individuals by cmv value to check cmv values 
cmvValues=dfOriginal['cmv'].value_counts().sort_index() 
cmvValues #There are values that are too high and values that are too low that might be considered as outliers.
# Create a boxplot to better visualize those values
plt.figure()
sb.boxplot(x=dfOriginal["cmv"]) 
plt.show()

# Create a new column for negative outliers that indicates outliers as 1 and other values as 0. Clients that give huge losses to the company will have value 1 in this column.
# When creating the column put the 6 lower values that are represented on the boxplot (outliers) with value 1.
dfOriginal['df_OutliersLow_cmv'] = np.where(dfOriginal['cmv']<=cmvValues.index[5],1,0)
dfOriginal['Others'] = np.where(dfOriginal['cmv']<=cmvValues.index[5],1,dfOriginal['Others'])
# Verify if the column was created as supposed
dfOriginal['df_OutliersLow_cmv'].value_counts()

# Create a box plot without the identified outliers:
plt.figure()
sb.boxplot(x = dfOriginal["cmv"][dfOriginal['df_OutliersLow_cmv'] == 0])
plt.show() 
#Check the ploted values in more detail:
cmvValues = dfOriginal['cmv'][dfOriginal['df_OutliersLow_cmv']==0].value_counts().sort_index()
cmvValues
# There are 6 lower values and 3 higher values that will be considered as outliers.

# Create a new column for positive outliers that indicates outliers as 1 and other values as 0. Clients that give huge profit to the company will have value 1 in this column.
# When creating this column put the 3 lower values that are represented on the boxplot (outliers) with value 1.
dfOriginal['df_OutliersHigh_cmv'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,0)
dfOriginal['Others'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,dfOriginal['Others'])
# Verify if the column was created as supposed
dfOriginal['df_OutliersHigh_cmv'].value_counts()

# Change the values of the new negative outliers to 1 in the df_OutliersLow_cmv column
dfOriginal['df_OutliersLow_cmv'] = np.where(dfOriginal['cmv']<=cmvValues.index[5],1,dfOriginal['df_OutliersLow_cmv'])
dfOriginal['Others'] = np.where(dfOriginal['cmv']<=cmvValues.index[5],1,dfOriginal['Others'])
# Verify if values were changed as supposed
dfOriginal['df_OutliersLow_cmv'].value_counts()

# Create a box plot without the until now identified outliers:
plt.figure()
sb.boxplot(x = dfOriginal["cmv"][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)]) 
plt.show()
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
plt.figure()
sb.boxplot(x = dfOriginal["cmv"][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersLow_cmv'] == 0)]) 
plt.show()
#Check the ploted values in more detail:
cmvValues = dfOriginal['cmv'][(dfOriginal['df_OutliersLow_cmv']==0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)].value_counts().sort_index()
cmvValues
#There are 3 higher values that will be considered as outliers.

# Change the values of the new positive outliers to 1 in the df_OutliersHigh_cmv column
dfOriginal['df_OutliersHigh_cmv'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,dfOriginal['df_OutliersHigh_cmv'])
dfOriginal['Others'] = np.where(dfOriginal['cmv']>=cmvValues.index[-3],1,dfOriginal['Others'])
# Verify if values were changed as supposed
dfOriginal['df_OutliersHigh_cmv'].value_counts()

# Create a box plot without the until now identified outliers:
plt.figure()
sb.boxplot(x = dfOriginal["cmv"][(dfOriginal['df_OutliersLow_cmv'] == 0) & (dfOriginal['df_OutliersHigh_cmv'] == 0)]) 
plt.show()

# 8. claims

plt.figure()
dfOriginal['claims'].value_counts().sort_index().plot() # there might be strange values on the claims that are distorting the plot.
plt.show()

plt.figure()
dfOriginal['claims'].hist()
plt.show()

# Check variable values:
valuesClaims = dfOriginal['claims'].value_counts().sort_index()
valuesClaims
# There is a small group of individuals who have high values of claims rate and that is the reason the plots are so distorted.

# Plot only values less than 3
dfClaims=dfOriginal.groupby(['claims'])['claims'].count()
dfClaims=pd.DataFrame(dfClaims, columns=['claims'])
dfClaims=dfClaims[dfClaims.index<3]
plt.figure()
dfClaims['claims'].sort_index().plot()
plt.show()
# People who have a claims rate of 0 are the ones with which the company did not spend anything.
# People who have a claims rate between 0 and 1 (excluding) are the ones with which the company had profit. This means that the amount paid by the company was less than the premiums paid by the clients.
# People who have a claims rate of 1 are the ones with which the company had no profit nor losses. 
# People who have a claims rate higher than 1 are the ones with which the company had losses.

#Lets look at people that have a claims rate lower than 1:
dfClaims=dfClaims[dfClaims.index<1]
dfClaims['claims'].sort_index().sum() #there are 8056 individuals that have a claims rate lower than 1
#Plot of the results
plt.figure()
dfClaims['claims'].sort_index().plot()
plt.show()

plt.figure()
dfOriginal['claims'][dfOriginal['claims']<3].hist()
plt.show()

# Lets distinguish between individuals that give losses (losses), individuals that give profit (profits), individuals that do not give profits nor losses (neutrals) and individuals with which the company did not spend anything (investigate). 
# The individuals that have a column value of 'investigate' need to be investigated later as we do not have any information about their premium values on this variable. We only know that the amount paid by the company is zero. We will need to study the premium value with the premium variables studied furtherahead.
dfOriginal['catClaims']=np.where(dfOriginal['claims']==0,'investigate','losses')
dfOriginal['catClaims']=np.where((dfOriginal['claims']>0)&(dfOriginal['claims']<1),'profits',dfOriginal['catClaims'])
dfOriginal['catClaims']=np.where((dfOriginal['claims']>0)&(dfOriginal['claims']==1),'neutrals',dfOriginal['catClaims'])
#Check if the new column was created as wanted
dfOriginal['catClaims'].value_counts()

# 9. lobMotor

# Plot lobMotor for a first visual analysis:
plt.figure()
dfOriginal['lobMotor'].value_counts().sort_index().plot() 
plt.show()

plt.figure()
dfOriginal['lobMotor'].hist() # There might be few high values that are distorting the graphs
plt.show()

# Check variable values:
valueslobMotor = dfOriginal['lobMotor'].value_counts().sort_index()
valueslobMotor

plt.figure()
sb.boxplot(x=dfOriginal["lobMotor"])
plt.show()

# Lets look for the fence high value of the box plot to define a from which value the lobMotor premium can be considered as an outlier.
q1=dfOriginal["lobMotor"].quantile(0.25)
q3=dfOriginal["lobMotor"].quantile(0.75)
iqr=q3-q1 #Interquartile range
fence_high=q3+1.5*iqr

# Create a column that indicates if an individual is outlier or not (if it is, the column value will be 1)
dfOriginal['Outliers_lobMot']=np.where(dfOriginal['lobMotor']>fence_high,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobMotor']>fence_high,1,dfOriginal['Others'])
# Verify if column was created correctly:
dfOriginal['Outliers_lobMot'].value_counts()

# Create a box plot without the outliers:
plt.figure()
sb.boxplot(x = dfOriginal['lobMotor'][dfOriginal['Outliers_lobMot']==0]) 
plt.show()
# 10. lobHousehold

# Plot lobHousehold for a first visual analysis:
plt.figure()
dfOriginal['lobHousehold'].value_counts().sort_index().plot() 
plt.show()

plt.figure()
dfOriginal['lobHousehold'].hist() # There might be few high values that are distorting the graphs
plt.show()

valueslobHousehold = dfOriginal['lobHousehold'].value_counts().sort_index()
valueslobHousehold

# Box plot 
plt.figure()
sb.boxplot(x=dfOriginal["lobHousehold"])
plt.show()

#Lets define 3000 from which individuals are considered outliers
# Create a column that indicates individuals that are outliers and individuals that are not (the column values will be 1 if the individuals are considered outliers)
dfOriginal['Outliers_lobHousehold']=np.where(dfOriginal['lobHousehold']>3000,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobHousehold']>3000,1,dfOriginal['Others'])
# Verify if column was created correctly:
dfOriginal['Outliers_lobHousehold'].value_counts()
plt.figure()
sb.boxplot(x = dfOriginal['lobHousehold'][dfOriginal['Outliers_lobHousehold']==0]) 
plt.show()

plt.figure()
dfOriginal['lobHousehold'][dfOriginal['Outliers_lobHousehold']==0].value_counts().sort_index().plot()
plt.show()
# We can observe that there are much more individuals with low values of household premiums than with high values, which makes sense because there are less houses that are expensive than cheaper ones.

# Transformar em Logaritmo? (ver mais tarde)

# 11. lobHealth 

plt.figure()
dfOriginal['lobHealth'].value_counts().sort_index().plot() 
plt.show()

plt.figure()
dfOriginal['lobHealth'].hist() # There might be few high values that are distorting the graphs
plt.show()

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
# Verify if column was created correctly:
dfOriginal['Outliers_lobHealth'].value_counts()

#Box plot without outliers
plt.figure()
sb.boxplot(x = dfOriginal['lobHealth'][dfOriginal['Outliers_lobHealth']==0]) 
plt.show()

plt.figure()
dfOriginal['lobHealth'][dfOriginal['Outliers_lobHealth']==0].value_counts().sort_index().plot()
plt.show()

# We can observe that health premiums follows an approximatelly normal distribution, which makes sense. 
# On one hand, the health premiums are not as expensive as, for example, the house hold premiums, so more people have access to it. 
# On the other hand, people consider health a primary preocupation and need. 
# There are more people that invest a medium value on health premiums. Then, there are people who invest less (poorer people) and people who invest more (richer people)

# 12. lobLife 

plt.figure()
dfOriginal['lobLife'].value_counts().sort_index().plot() 
plt.show()

plt.figure()
dfOriginal['lobLife'].hist()
plt.show()

# Check variable values:
valueslobLife  = dfOriginal['lobLife'].value_counts().sort_index()
valueslobLife 

# Box plot
plt.figure()
sb.boxplot(x = dfOriginal['lobLife']) 
plt.show()
# We decided not to consider outliers on this variable as there are no extreme individuals that influence the distribution (no individuals that highlight over the others).

# We can observe that more people invest less on life premiums.

# Transformar em Logaritmo? (ver mais tarde)

# 13. lobWork  
plt.figure()
dfOriginal['lobWork'].value_counts().sort_index().plot() 
plt.show()

plt.figure()
dfOriginal['lobWork'].hist()# There might be few high values that are distorting the graphs
plt.show()

# Check variable values:
valueslobWork  = dfOriginal['lobWork'].value_counts().sort_index()
valueslobWork 

#Box plot
plt.figure()
sb.boxplot(x = dfOriginal['lobWork']) 
plt.show()
#Lets define 400 from which individuals are considered outliers
# Create a column that indicates individuals that are outliers and individuals that are not (the column values will be 1 if the individuals are considered outliers)
dfOriginal['Outliers_lobWork']=np.where(dfOriginal['lobWork']>400,1,0)
dfOriginal['Others']=np.where(dfOriginal['lobWork']>400,1,dfOriginal['Others'])
# Verify if column was created correctly:
dfOriginal['Outliers_lobWork'].value_counts()

#Box plot without outliers
plt.figure()
sb.boxplot(x = dfOriginal['lobWork'][dfOriginal['Outliers_lobWork']==0]) 
plt.show()

plt.figure()
dfOriginal['lobWork'][dfOriginal['Outliers_lobWork']==0].value_counts().sort_index().plot()
plt.show()

# There is a high number of indiviaduals with low work premiums values.

# Transformar em Logaritmo? (ver mais tarde)

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
#There are 1997 people that have a firt policy before even being born. This does not make any sense.

# Create age column that calculates current ages of customers (useful to check the next incoherence).
dfOriginal['age']= 2016-dfOriginal['birthday']
# Check if there are people with less than 16 years old (including) who have children 
dfOriginal['children'][dfOriginal['age']<=16].value_counts()
# There are 31 people who are younger or equal to 16 years old and that have children and 16 younger or equal to 16 years old and that do not have children.
# At this age, in normal situations, there should be zero people with children. Even if there were some cases in these situations, 31 is a huge number.

# Check if people with age <9 years old have basic education.
dfOriginal['education'][dfOriginal['age']<9].value_counts()
dfOriginal[dfOriginal['age']<9] # There are no people with less than 9 years old.

# Check if people with age <16 years old have high school education. This would not make sense: in normal circumstances, people with less than 16 years old have not completed the high school education yet.
dfOriginal['education'][dfOriginal['age']<16].value_counts()
# People with less than 16 years old only have basic education (12 people), which makes sense.

# Check if people with age <20 years old have Bsc/Msc education. This would not make sense: in normal circumstances, people with less than 20 years old have not completed the Bsc/ Msc education yet.
dfOriginal['education'][dfOriginal['age']<20].value_counts()
# People with less than 20 years old only have basic education (262 people) and high school education (81 people).

# Check if people with age <25 years old have Phd education. This would not make sense: in normal circumstances, people with less than 25 years old have not completed the Phd education yet.
dfOriginal['education'][dfOriginal['age']<25].value_counts()
#There are 8 people who have a Phd with age less than 25 years old, which does not make sense.

# Check if people with less than 16 years old have a salary>0
dfOriginal['salary'][(dfOriginal['age']<16) & (dfOriginal['salary']>0)].count()
#there are 12 people with less than 16 years old and that have a salary, which does not make sense. At these ages the expected salary value was expected to be zero, which means that we were expecting the output of this code line to be zero.

# Check if people with less than 16 years old have a motor premium. This would not make sense as people with these ages do not have driving license.
dfOriginal['lobMotor'][dfOriginal['age']<16].count()
# There are 12 people with less than 16 years old that have a motor premium, which does not make sense.

# Check if people with less than 18 years old have a household premium (we defined the age 18 years old as the minimum age for a person to get a house).
dfOriginal['lobHousehold'][dfOriginal['age']<18].count()
# There are 116 people younger than 18 years old who have a household premium, which does not make sense.

# Check if people with less than 16 years old have a work compensation premium, which does not make sense. The minimum age to start working is 16 years old.
dfOriginal['lobWork'][dfOriginal['age']<16].count()
# There are 12 people younger than 16 years old that have a work compensation premium.

# Final Decision: drop birthday and age columns - they do not make any sense when considering other variables in the data set.

# Create a column for year salary (useful to check the next incoherence).
dfOriginal['yearSalary']=dfOriginal['salary']*12

# Create a column with the total premiums (useful to check the next incoherence)
dfOriginal['lobTotal']=dfOriginal['lobMotor']+dfOriginal['lobHousehold']+dfOriginal['lobHealth']+dfOriginal['lobLife']+dfOriginal['lobWork']

# Check if the 30% of the year salary is higher than the lobTotal
dfOriginal[dfOriginal['yearSalary']*0.3<dfOriginal['lobTotal']]
# There are 14 people that spend more than 30% of the year salary in the total of premiums.

# Check if the 50% of the year salary is higher than the lobTotal
dfOriginal['id'][dfOriginal['yearSalary']*0.5<dfOriginal['lobTotal']].count()
# There are 2 people that spend more than 50% of the year salary in the total of premiums, which migh be considered strange. It is not normal for a person spending more than 50% of the salary in premiums.

# Check if the year salary is higher than the lobTotal. If not, it does not make sense.
dfOriginal['id'][dfOriginal['yearSalary']<dfOriginal['lobTotal']].count()
# There is one person that has a salary lower than the lobTotal, which does not make sense.
# We decided to add this customer to an incoherence column.
dfOriginal['incoherences']=np.where(dfOriginal['yearSalary']<dfOriginal['lobTotal'],1,0)
dfOriginal['Others']=np.where(dfOriginal['yearSalary']<dfOriginal['lobTotal'],1,dfOriginal['Others'])

#---------------CREATE A DATA FRAME TO WORK ON (WITH NO INCOHERENCES AND NO OUTLIERS)------------------------
# Take outliers, strange values and incoherences out:
dfWork=dfOriginal[dfOriginal['Others']==0]

# Drop columns that are not needed: because have all values zero and other reasons
dfWork=dfWork.drop(columns=['age', 'birthday','incoherences','Strange_birthday','Others','strange_firstPolicy','Outliers_salary','df_OutliersLow_cmv','df_OutliersHigh_cmv','Outliers_lobMot','Outliers_lobHousehold','Outliers_lobHealth','Outliers_lobWork'])

#----------------------------------MISSING VALUES----------------------------

# Create a column called 'Nan' to count the number of missing values by row
dfNan=dfWork.drop(columns=['yearSalary','lobTotal'])
dfNan['Nan']=dfNan.isnull().sum(axis=1)

# Check the number of rows that have 0, 1, 2, 3 etc Nan values
dfNan['Nan'].value_counts()
# Maximum number of Nan values is a row is 3.

# Count missing values by column:
dfNan.isnull().sum()

# LOBs
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
#check missing values by row:
dfNan['Nan'].value_counts()

# Recalculate the column lobTotal (as there are no Null values on the lob variables anymore)
dfWork['lobTotal']=dfWork['lobMotor']+dfWork['lobHousehold']+dfWork['lobHealth']+dfWork['lobLife']+dfWork['lobWork']
# Check if lobTotal does not have Nan values
dfWork.isnull().sum()

#Check again Nan values per column:
dfNan.isnull().sum()

# LIVING AREA
# In living area there is one null value. Lets try to check and treat it

# This is the row:
null=dfNan[dfNan['livingArea'].isnull()]
null

#Graph of lobhousehold and living area:
plt.figure()
sb.barplot(x="livingArea",y="lobHousehold",data=dfNan)
plt.show()

# Check lobHousehold (>=0) by living area:
dfNan[dfNan['lobHousehold']>=0].groupby(by=["livingArea"])["lobHousehold"].mean().reset_index(name = 'Average_lobH')
# The average of lobHousehold does not differentiate a lot between different living areas.
dfNan[dfNan['lobHousehold']>=0].groupby(by=["livingArea"])["lobHousehold"].var().reset_index(name = 'Var_lobH')
# The variance does not differ a lot between living areas 1, 2 and 3, but the 4th one differs a lot from these 3 first areas.

# Living area might be determined by LobHousehold
plt.figure()
sb.boxplot(x='livingArea', y='lobHousehold', data=dfWork)
plt.show()
# The boxplot shows that this does not happen

# Treat the Nan value through KNN
# Which variables should we use to predict the lobHousehold?
dfWork2=dfWork.dropna()

plt.figure()
sb.pairplot(dfWork2, vars=['firstPolicy','salary','cmv','claims','lobHousehold'], hue='livingArea')
plt.show()
# There is no variable that explains the variable living area
plt.figure()
sb.pairplot(dfWork2, vars=['lobMotor','lobHealth','lobLife','lobWork','lobTotal','yearSalary'], hue='livingArea')
plt.show()
# There is no variable that explains the variable living area

plt.figure()
sb.pairplot(dfWork2, vars=['children','catClaims'], hue='livingArea')
plt.show()


dfWork2.groupby(by='livingArea').hist(alpha=0.4)
# We realized that the variable livingArea is not explained by any of the other variables. Besides, we do not have any information about this variable's categories. 

############################################################################################3
# CHILDREN - MISSING VALUES IMPUTATION
from sklearn.impute import KNNImputer

#Check null values on children:
dfWork['children'].isna().sum()          #There are 21 Nan values

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





###############################################################################################

# Create a data frame that has Nan values on children
children_incomplete=dfWork2.loc[dfWork2.children.isna(),]

# Create a data frame that has no Nan values on children
children_complete=dfWork2[~dfWork2.index.isin(children_incomplete.index)]

#To do a classifier we need to change it to string
children_complete.children=children_complete.children.astype('str')

clf = KNeighborsClassifier(5,weights='distance', metric='euclidean')

# Use the children_complete data frame to train the model:

trained_model = clf.fit(children_complete.loc[:,['salary', 'lobMotor']],
                        children_complete.loc[:,'children'])

#Drop children on the incomplete data set
imputed_values = trained_model.predict(children_incomplete.drop(columns=['children']))

#reshape: -1 I dont know how many rows
temp_df = pd.DataFrame(imputed_values.reshape(-1,1), columns = ['children'])

#drop children column from children_incomplete data frame:
children_incomplete =children_incomplete.drop(columns=['children'])

my_data_incomplete =my_data_incomplete.reset_index(drop=True)

#concat to put everything together: concat the my_data_incomplete data frame (that does not have dependents column) with the temp_df (that has the dependents column determined by the KNN):
my_data_incomplete =pd.concat([my_data_incomplete, temp_df],
                              axis=1,
                              ignore_index=True,
                              verify_integrity=False)

my_data_incomplete.columns = ['age','income', 'frq', 'dependents']
# Missing the step of joining both the incomplete and the complete data frames again.

















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
corrMatrix=dfCorr.corr(method ='pearson')

plt.figure()
fig, ax = plt.subplots(figsize=(10,10))
sb.heatmap(corrMatrix,annot=True,fmt=".3f")
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




