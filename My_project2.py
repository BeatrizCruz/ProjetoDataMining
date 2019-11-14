# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:44:55 2019

@author: aSUS
"""
#OLA
import pandas as pd
import sqlite3
import numpy as np
from matplotlib import pyplot as plt
import math
import seaborn as sns


my_path = 'C:/Users/aSUS/Documents/IMS/Master Data Science and Advanced Analytics with major in BA/Data mining/Projeto/insurance.db'
#dbname = "datamining.db"

# connect to the database
conn = sqlite3.connect(my_path)
#cursor = conn.cursor()
conn.row_factory=sqlite3.Row

#tables in the data base:
query = "select name from sqlite_master where type='table'"
df2 = pd.read_sql_query(query,conn)

#Columns in each table of the data base:
query2="select sql from sqlite_master where tbl_name='LOB' and type='table'"
print(pd.read_sql_query(query2,conn))

cur.execute('select * from LOB')
col_name_list=[tuple[0] for tuple in cur.description]

query="""
select * from lob limit(10);
"""

query="""
select * from engage limit(10);
"""
 
my_table= cursor.execute(query).fetchall()
cursor.execute("select name from sqlite_master where type='table'")
print(cursor.fetchall())

#Diretorias:
file='C:/Users/aSUS/Documents/IMS/Master Data Science and Advanced Analytics with major in BA\Data mining/Projeto/A2Z Insurance.csv'

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

#--------------------STUDY OF VARIABLES-----------------------
# 1. birthday:

dfOriginal['birthday'].value_counts().sort_index().plot() # there is a strange value on the birthday that is distorting the plot.
dfOriginal['birthday'].hist() # The plot with the strange value is not perceptive

# Check variable values:
dfOriginal['birthday'].value_counts().sort_index() # strange value on the birthday: 1028

# Create a new data frame with strange values on birthday:

dfOriginal['Strange_birthday']=np.where(dfOriginal['birthday']<1900, '1','0')
dfOriginal['Strange_birthday'].value_counts()

#Plot birthday variable:

dfOriginal['birthday'][dfOriginal['Strange_birthday']=='0'].hist()
dfBirthday['birthday'][dfOriginal['Strange_birthday']=='0'].value_counts().sort_index().plot(marker='o')
plt.show()

#2. firstPolicy

dfOriginal['firstPolicy'].value_counts().sort_index().plot() # there is a strange value on firstPolicy that is distorting the plot.
dfOriginal['firstPolicy'].hist() # The plot with strange values is not perceptive

#Check strange values
dfOriginal['firstPolicy'].value_counts().sort_index() # there is a strange value: 53784

# create a new data frame with strange values on birthday:

dfOriginal['strange_firstPolicy']=np.where(dfOriginal['firstPolicy']>2016, '1','0')
dfOriginal['strange_firstPolicy'].value_counts()

#Plot birthday variable:
dfOriginal['firstPolicy'][dfOriginal['strange_firstPolicy']=='0'].hist()
dfOriginal['firstPolicy'][dfOriginal['strange_firstPolicy']=='0'].value_counts().sort_index().plot(marker='o')
plt.show()

# 3. education

# Create a variable to count the individuals per category:
counteducation=dfOriginal['education'].value_counts().sort_index()
plt.bar(np.arange(len(count.index)),count)
plt.xticks(np.arange(len(count.index)),count.index)
plt.show()
# There is a considerable number of individuals with high education (BSc/MSc and PhD). 
#The number of individuals having PhD is not that high. We will consider later if it makes sense to join the categories BSc/MSc and PhD in a unique category.

# 4. salary
# To study this variable as it has different values that are not easily repeated through individuals, instead of counting by value as done with the previous cases, we decided to make the cumulative to plot.
dfOriginal['salary'].value_counts().sort_index().plot() # there is a strange value on firstPolicy that is distorting the plot.
dfOriginal['salary'].hist() # The plot with the strange value is not perceptive

# Create a new data frame with outlier values on salary
dfOriginal['Outliers_salary']=np.where(dfOriginal['salary']>10000, '1','0')
dfOriginal['Outliers_salary'].value_counts()


# Plot the salary values and the cumulative values of salary
countSalary = dfOriginal['salary'].value_counts().sort_index() #there are 2 out of the box values: 34490, 55215
countSalaryCum = countSalary.cumsum()
countSalaryCum.plot()


dfOriginal['salary'][dfOriginal['Outliers_salary']=='0'].hist()
dfOriginal['salary'][dfOriginal['Outliers_salary']=='0'].value_counts().sort_index().plot(marker='o')



dfOriginal['salary'][dfOriginal['Outliers_salary']=='0'].value_counts().sort_index().cumsum().plot(marker='o')






# Check with log dist: as the usual behavior of a salary variable is a distribution with a heavy tail on the left side, usually it is applied a log transformation on the distribution in order to transform it to a normal distribution.
dfOriginal['logSalary'] = np.log(dfOriginal['salary'])
countLogSalary=dfOriginal['logSalary'].value_counts().sort_index()
countLogSalaryCum= countLogSalary.cumsum()
countLogSalary.plot()
countLogSalaryCum.plot()
dfOriginal=dfOriginal.drop(['logSalary'], axis=1)
# Log distributon: not applicable as original distribution is already normal. It does not follow the usual behavior. 

# 5. livingArea

# Create a variable to count the individuals per category:
countlivingArea=dfOriginal['livingArea'].value_counts().sort_index()
countlivingArea

#Create a bar chart that shows the number of individuals per living area
plt.bar(np.arange(len(countlivingArea.index)),countlivingArea)
plt.xticks(np.arange(len(countlivingArea.index)),countlivingArea.index)
plt.show()

# 6. children

# Create a variable to count the individuals per category:
countchildren=dfOriginal['children'].value_counts().sort_index()
countlivingArea
#Create a bar chart that shows the number of individuals per living area
plt.bar(np.arange(len(countchildren.index)),countchildren)
plt.xticks(np.arange(len(countchildren.index)),countchildren.index)
plt.show()
#There are more individuals with children that without.

# 7. cmv 

# To study this variable as it has different values that are not easily repeated through individuals, instead of counting by value, we decided to make the cumulative to plot, as done with the salary variable.
dfOriginal['cmv'].value_counts().sort_index().plot() # there is a strange value on firstPolicy that is distorting the plot.
dfOriginal['cmv'].hist() # The plot with the strange value is not perceptive

cmvValues=dfOriginal['cmv'].value_counts().sort_index() #There are values that are too high and values that are too low that might be considered as outliers.
dfOriginal.dtypes
sns.boxplot(x=dfOriginal["cmv"]) 

# Take the 6 lower values that are represented on the boxplot (outliers) - clients that give losses
df_OutliersLow_cmv = dfOriginal.loc[(dfOriginal['cmv']<=cmvValues.index[5])]
df_Outliers_cmv['cmv']

# Change the original data frame, taking the strange values away
dfcmv=dfOriginal.loc[~(dfOriginal['cmv']<=cmvValues.index[5])]
sns.boxplot(x=dfcmv["cmv"]) 

# Take out lower values and higher values
cmvValues=dfcmv['cmv'].value_counts().sort_index()

df_OutliersLow_cmv = df_OutliersLow_cmv.append(dfcmv.loc[(dfcmv['cmv']<=cmvValues.index[5])])
df_OutliersHigh_cmv= dfcmv.loc[(dfcmv['cmv']>=cmvValues.index[-3])]
dfcmv=dfcmv.loc[~(dfcmv['cmv']<=cmvValues.index[5])]
dfcmv=dfcmv.loc[~(dfcmv['cmv']>=cmvValues.index[-3])]
sns.boxplot(x=dfcmv["cmv"]) 

df_OutliersLow_cmv = df_OutliersLow_cmv.append(dfcmv.loc[(dfcmv['cmv']<=cmvValues.index[1])])
dfcmv=dfcmv.loc[~(dfcmv['cmv']<=cmvValues.index[1])]
sns.boxplot(x=dfcmv["cmv"]) 


countCmv=dfOriginal['cmv'].sort_index()
countCmvCum= countCmv.cumsum()
countCmv.plot()
countCmvCum.plot()

# colunas 0 e 1 ( outliers e strange values)
#


















#-----------------CHECK INCOHERENCES------------------#

#if birthday is higher than First policy's year: 
agesList=[18,16,15,10,5,0]
countList=[]
for j in agesList:
    count_inc=0
    for i, index in dfOriginal.iterrows():
        if dfOriginal.loc[i,'firstPolicy']-dfOriginal.loc[i,'birthday']<j: count_inc+=1
    countList.append(count_inc)
countList


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
import seaborn as sns
sns.boxplot(x=dfOriginal["birthday"]) 
sns.boxplot(x=dfOriginal["education"]) 
sns.boxplot(x=dfOriginal["salary"]) 
sns.boxplot(x=dfOriginal["livingArea"]) 
sns.boxplot(x=dfOriginal["children"]) 
sns.boxplot(x=dfOriginal["cmv"]) 
sns.boxplot(x=dfOriginal["claims"]) 
sns.boxplot(x=dfOriginal["lobMotor"]) 
sns.boxplot(x=dfOriginal["lobHousehold"]) 
sns.boxplot(x=dfOriginal["lobHealth"]) 
sns.boxplot(x=dfOriginal["lobLife"])
sns.boxplot(x=dfOriginal["lobWork"]) 

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




