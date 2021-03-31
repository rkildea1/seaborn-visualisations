import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
import numpy as np
#import the file as a pandas df 
#file comes from Kaggle
stalcDF = pd.read_csv("student-mat.csv", encoding = "ISO-8859-1")


#playing around with some encoding.. 

#first thing to do is pair up the data headers with the data in the first row
ob = stalcDF.iloc[0].reset_index().apply(tuple, axis=1)  #write array col head & row1 to a dict_items
                                       
colHeader_row1_list = [] 
for key, value in ob:
    list_item = (key,value)
    colHeader_row1_list.append(list_item)

categorical_cols = [] #this will be the list of categorical attributes
numeric_cols = []     #this will be the list of numeric attributes

for i, x in colHeader_row1_list:
#     print((type(x)),i,x) # just checking what kind of format the data is in (e.g., int or numpy.int64)
    if type(x) != str: 
        numeric_cols.append(i)
    else:
        categorical_cols.append(i)
        
print("There are:", (len(categorical_cols)), "text columns")
                            # 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                            # 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                            # 'nursery', 'higher', 'internet', 'romantic']
print("There are:", (len(numeric_cols)), "numeric columns")
                            # ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                            #  'failures', 'famrel', 'freetime', 
                            #  'goout', 'Dalc', 'Walc', 'health', 
                            #  'absences', 'G1', 'G2', 'G3']
                
          
#convert text in the columns with text data to numeric form via label encoding..
#which adds 17 new features to my set.
for i in categorical_cols:
#     print(i)
    stalcDF[i] = stalcDF[i].astype("category")
    stalcDF[(i+"_CAT")]  = stalcDF[i].cat.codes #duplicate each column encoded, and add "_CAT" after it
#len(stalcDF.columns) # prints = 50
        
# #duplicate the DataFrame
stalcDF_N= stalcDF.copy() 

#Use this next section if you want to drop the columns with text data (i commented it out)

# print("Length of stalcDF_N before dropping textual columns:",len(stalcDF_N.columns)) #should print 50 on the first run
# for i in categorical_cols:
#     if i in stalcDF_N:
#         #print("dropping", i, "\n....")
#         stalcDF_N.drop([i], axis=1, inplace=True)
#         print("dropped", i, "!")
#     else:
#         pass
# print("\nLength of stalcDF_N after dropping textual columns:",len(stalcDF_N.columns)) #should print 31

# #Len of new DF should be 33: 50 -17 (the len of textual columns identifed earlier)          



#creating some new colums for some simple visual overviews 
stalcDF_N['Avg_Grade'] = stalcDF_N[['G1', 'G2','G3']].mean(axis=1) 
stalcDF_N['Avg_Alc_Cnsmptn'] = stalcDF_N[['Dalc', 'Walc']].mean(axis=1)
#Could come back and drop these columns or their base columns, if applying machine learning.
           
  
#Seaborn!!!!!!

#You can find out more information here if you can't follow the below https://www.youtube.com/watch?v=Pkvdc2Z6eBg&ab_channel=Data360YP


#######  
  #Lets add some different styles of plots (bar/ hist/ scatter etc.,.)
#######
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Provide an overall title')
fig = fig.tight_layout(pad=3)

sns.lineplot(ax = axes[0,0], 
             x = "studytime",
             y = "Avg_Grade", 
             data = stalcDF_N, 
             hue = 'sex', 
             style = 'sex', 
             markers =True) 

sns.boxplot(ax = axes[0,1], 
            x = "Avg_Alc_Cnsmptn",
            y = "age",
            data = stalcDF_N)

sns.barplot(ax = axes[0,2], 
            x = "traveltime",
            y = "Avg_Alc_Cnsmptn",
            hue = "address",
            ci = False,
            data = stalcDF_N)

sns.barplot(ax = axes[1,0], 
            x = "Avg_Alc_Cnsmptn",
            y = "traveltime",
            hue = "address",
            ci = False,
            data = stalcDF_N,
            orient = 'h',
            color = '#11E5AD') #hex codes
    
sns.boxplot(ax = axes[1,1], 
            x = "age",
            data = stalcDF_N,
            )

sns.histplot(ax = axes[1,2], 
             x = "Avg_Grade",
             kde=True,
             data = stalcDF_N,
             hue='guardian',
             bins=15)
          
  
#######  
  #Lets add some box plots
#######
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle('Mother and Fathers Jobs / Average Grade ')
fig = fig.tight_layout(pad=3)

                                                                    # Fathers Job

sns.boxplot(ax = axes[0],
            x = "Fjob",       # x variable name
            y = "Avg_Grade",       # y variable name
            data = stalcDF_N,     # dataframe to plot
            color = "#ebe310")
sns.swarmplot(ax = axes[0],
            x = "Fjob",       # x variable name
            y = "Avg_Grade",       # y variable name
            data = stalcDF_N,     # dataframe to plot
            color = "#4a9dc7")

                                                                    # Mothers Job
sns.boxplot(ax = axes[1],
            x = "Mjob",       # x variable name
            y = "Avg_Grade",       # y variable name
            data = stalcDF_N,     # dataframe to plot
            color = "#ebe310")
sns.swarmplot(ax = axes[1],
            x = "Mjob",       # x variable name
            y = "Avg_Grade",       # y variable name
            data = stalcDF_N,     # dataframe to plot
            color = "#4a9dc7")




#######  
  #Lets add some more variations of plots
#######

fig, axes = plt.subplots(2, 2, figsize=(15, 8))
fig.suptitle('Variety of graphs')
fig = fig.tight_layout(pad=3)

                                                                    # 

sns.scatterplot(ax = axes[0,0],
            x = "absences",       # x variable name
            y = "Avg_Alc_Cnsmptn",       # y variable name
            hue = "guardian",  # group variable name
            data = stalcDF_N,     # dataframe to plot
            color = "#ebe310")
                                                         # 
sns.boxplot(ax = axes[0,1],
            x = "absences",       # x variable name
            y = "Avg_Alc_Cnsmptn",       # y variable name
#             hue = "sex",  # group variable name
            data = stalcDF_N,     # dataframe to plot
            color = "#ebe310")

sns.histplot(ax = axes[1,0],x = "Avg_Grade",kde=True,data = stalcDF_N,hue='guardian',bins=15)
sns.barplot(ax = axes[1,1],x="age", y='Avg_Grade',ci="sd", palette="dark", alpha=.6, data = stalcDF_N)

#######  
  #Lets add some graphs without grids
#######

#tryig without a grid of graphs

sns.catplot(
            x = "guardian",       # x variable name
            y = "Avg_Grade",       # y variable name
            hue = "sex",  # group variable name
            data = stalcDF_N,     # dataframe to plot
            kind = "bar",
            color = "green"
            )


sns.catplot(
            x = "guardian",       # x variable name
            y = "Avg_Grade",       # y variable name
            hue = "sex",  # group variable name
            data = stalcDF_N,# dataframe to plot
            kind="violin",
            col="higher",
            color = "orange"
            )

#######  
  #Lets add some graphs through the pairplot. I personally find these are handy when you select a smaller number of features
#######

#a reduced size set analysed.. 
stalcDF_N4 = stalcDF_N[["age","Avg_Grade","Avg_Alc_Cnsmptn","higher_CAT"]]
stalcDF_N4.head()
sns.set()
sns.pairplot(stalcDF_N4, hue='higher_CAT')
