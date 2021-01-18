
# I have a train data file(Train_data1.csv) with all the keywords to define a category.
# I want to find out top 5 co-related categories based on the description given in the test file (test_data_new.csv)

# how much close the other categories (top 5) are (listed in train_data1 : column C (Categories) with respect to the predicted Category.

# Wanted to check the description in the test_data_new.csv : how much its co-related/ close with other
#categories. Wanted  with to take top 5. The most significant is already predicted but other
# than the highest accurate prediction , for each other category , how its co-relation % is ?
# currently I am getting almost 90% accuracy in predicting the most closest category (only 1 category).
# is there any way/model/function  that I can use which is similar and predicting the top 5 Categories closest to
# the Description ( test_data_new.csv ) and the Keywords mention in the Train_data1 (Description, Category). If you think
#this can be achieved by looping through the category list and for each category separately calculating but here the predicion
# model only gives the closest one and not any other categories. I am not sure if its possible or not .


#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec 
import string
import nltk
from collections import Counter
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import csv
import numpy as np



#ps = PorterStemmer()


#df_train= pd.read_csv('sas_data_train.csv')

df_train= pd.read_csv('train_data1.csv',encoding='latin1')

#df_test = pd.read_csv('sas_data_test.csv')
df_test = pd.read_csv('test_data_new.csv',encoding='latin1')




#df_train.head(2) # we need 2 columns only - Description and Category - independent variable is Description and dependent variable is category


df_train_new = df_train.iloc[:,1:3]
#df_train_new.head(2)

#df_test.head(2) 


df_test_new = df_test.iloc[:,1:]
df_test_new.head(2) 

# unique categories in training set
categories = df_train['Category'].unique()
#print (categories)

# count of categories
df_train['Category'].count()


# Records in each category
df_train['Category'].value_counts()


# Health and Wellness have 12 records and then sports have 6 and so on.


x = df_train_new['Description']  # independent variable
y = df_train_new['Category']     # dependent variable

#x.head()

# Text cleaning required to delete stop words, puntuations
def text_cleaning(a):
     a = [w.lower() for w in a]    
     remove_punctuation = [char for char in a if char not in string.punctuation]
     remove_punctuation = ''.join(remove_punctuation)
     return [word for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]


from sklearn.feature_extraction.text import CountVectorizer


bow_transformer=CountVectorizer(analyzer=text_cleaning)
bow_transformer = bow_transformer.fit(df_train_new['Description']) 


print(len(bow_transformer.vocabulary_))   
#print(bow_transformer.vocabulary_)


# we have 2497 different vocabulary in our training dataset. And each word is converted into
#number because our algorithm understand data in the form of number. Unique number is
#assigned to every word. 




tokens = bow_transformer.get_feature_names()
#print(tokens)   



bow_transformer_train = bow_transformer.transform(df_train_new['Description'])
bow_transformer_train




X = bow_transformer_train.toarray()
#print(X)

 
print(X.shape) 
#we have 66 rows and every row have 2497 columns- 2497 columns because we have 2497 unique words present in our dataset


# If you want you can store it in a dataframe also
dd=pd.DataFrame(data=bow_transformer_train.toarray(),columns=tokens)
dd


df_test_new.head(2) 


y_test = df_test_new.loc[:,'Category']
#print(y_test.head(2))

#print (df_test_new['Description'].head(2))

cleaned_desc_test = df_test_new.loc[:,'Description'].apply(text_cleaning)

#print (cleaned_desc_test)

bow_transformer_test = bow_transformer.transform(df_test_new['Description'])
#bow_transformer_test = bow_transformer.transform(cleaned_desc_test)

#print(bow_transformer_test)

x_test = bow_transformer_test.toarray()
#print(x_test)

 
#print(x_test.shape)  


dd=pd.DataFrame(data=bow_transformer_test.toarray(),columns=tokens)
#print(dd.head())  # here you can see that the vocabulary size is 2497 vocabulary


# Now let's use Multinomial naive bayes algorithm. We will train our algorithm with
#train file and then test on test file



from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X,y)


prediction = model.predict(x_test)
#print(prediction)  



from sklearn.metrics import accuracy_score
#Printing accuracy of the our model
print(accuracy_score(prediction,y_test))


arr = np.array(prediction)

df_pred = pd.DataFrame(data=arr.flatten(),columns=['Model_Output'])
#print(df_pred)

out_desc_df = df_test_new['Description']

frames = [out_desc_df,df_pred]

out_row = pd.concat(frames,axis=1)

print (out_row)



out_row.to_csv (r'model_output.csv', index = False, header=True)


#df_cat_list = pd.read_csv('category_list.csv',encoding='latin1')
    

#category_list = df_cat_list['Category'].unique()
             
#print ("Full Category list" , category_list)














     



