# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:21:15 2023

@author: kailas
"""
######################################################################################
Problem statement:-
Build a recommender system by using cosine simillarties score.



#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#Dataset
data=pd.read_csv("D:/in class files/Movie 9-3.csv")

#EDA
data.head()
data.tail()
data.shape
data.info()
data.describe()
data.isnull().sum()



#number of unique users in the dataset
len(data.userId.unique())
len(data.movie.unique())

# Pivot table.
user_movies_data = data.pivot(index='userId',
                                 columns='movie',
                                 values='rating').reset_index(drop=True)

#Impute those NaNs with 0 values
user_movies_data.fillna(0, inplace=True)
user_movies_data

#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


user_sim = 1 - pairwise_distances( user_movies_data.values,metric='cosine')
user_sim


#Store the results in a dataframe
user_sim_data = pd.DataFrame(user_sim)


#Set the index and column names to user ids 
user_sim_data.index = data.userId.unique()
user_sim_data.columns = data.userId.unique()

user_sim_data.iloc[0:5, 0:5]

np.fill_diagonal(user_sim, 0)
user_sim_data.iloc[0:5, 0:5]

#Most Similar Users
user_sim_data.idxmax(axis=1)[0:5]

data[(data['userId']==6) | (data['userId']==168)]


user_1=data[data['userId']==6]
user_2=data[data['userId']==11]
user_2


user_1.movie

#Merge 
pd.merge(user_1,user_2,on='movie',how='outer')
