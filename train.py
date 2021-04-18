import pandas as pd, numpy as np, nltk, re, flask, json, html2text, pickle, joblib
from time import time
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.simplefilter("ignore")
from time import time
from multiprocessing import Process
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import 

new_data = pd.read_csv("./datasets/recommend_data.csv")
new_df=pd.DataFrame({'Headline':[],'region':[]})
for i,j in zip(new_data.article_headline, new_data.user_region):        
    k=j.split(',')        
    for b in range(len(k)):
        new_df=new_df.append({'Headline': i,'region':k[b]},ignore_index=True)
p=[]
for i in new_df.region.values:
    p.append(re.sub(r'[^a-zA-Z0-9_,]+', '', str(i)))
new_df.region=p


#mapping labels with values
mapping = {'Global_Edition':'0','Pacific':'1', 'South_Asia':'2',  'East_and_South_East_Asia':'3', 'Europe_and_Central_Asia':'4', 'Central_Africa':'5', 'East_Africa':'6',  'Southern_Africa':'7', 'Middle_East_and_North_Africa':'8','North_America':'9','Latin_America_and_Caribbean':'10','West_Africa':'11'}
new_df=new_df.replace({'region': mapping})
grouped = new_df.groupby(new_df.region)


Global_Edition = grouped.get_group('0')
Pacific = grouped.get_group('1')
South_Asia= grouped.get_group('2')
East_and_South_East_Asia= grouped.get_group('3')
Europe_and_Central_Asia= grouped.get_group('4')
Central_Africa= grouped.get_group('5')
East_Africa = grouped.get_group('6')
Southern_Africa= grouped.get_group('7')
Middle_East_and_North_Africa = grouped.get_group('8')
North_America= grouped.get_group('9')
Latin_America_and_Caribbean = grouped.get_group('10')
West_Africa= grouped.get_group('11')

Global_Edition.to_csv('Global_Edition.csv', index = None)
Pacific.to_csv('Pacific.csv', index = None)
South_Asia.to_csv('South_Asia.csv', index = None)
East_and_South_East_Asia.to_csv('East_and_South_East_Asia.csv', index = None)
Europe_and_Central_Asia.to_csv('Europe_and_Central_Asia.csv', index = None)
Central_Africa.to_csv('Central_Africa.csv', index = None)
East_Africa.to_csv('East_Africa.csv', index = None)
Southern_Africa.to_csv('Southern_Africa.csv', index = None)
Middle_East_and_North_Africa.to_csv('Middle_East_and_North_Africa.csv', index = None)
North_America.to_csv('North_America.csv', index = None)
Latin_America_and_Caribbean.to_csv('Latin_America_and_Caribbean.csv', index = None)
West_Africa.to_csv('West_Africa.csv', index = None)

data_0=pd.read_csv('Global_Edition.csv')
data_1=pd.read_csv('Pacific.csv')
data_2=pd.read_csv('South_Asia.csv')
data_3=pd.read_csv('East_and_South_East_Asia.csv')
data_4=pd.read_csv('Europe_and_Central_Asia.csv')
data_5=pd.read_csv('Central_Africa.csv')
data_6=pd.read_csv('East_Africa.csv')
data_7=pd.read_csv('Southern_Africa.csv')
data_8=pd.read_csv('Middle_East_and_North_Africa.csv')
data_9=pd.read_csv('North_America.csv')
data_10=pd.read_csv('Latin_America_and_Caribbean.csv')
data_11=pd.read_csv('West_Africa.csv')

global_edition=[]
for i in data_0.Headline.values:
    global_edition.append(str(i))
documents_0=global_edition

pacific=[]
for i in data_1.Headline.values:
    pacific.append(str(i))
documents_1=pacific

south_asia=[]
for i in data_2.Headline.values:
    south_asia.append(str(i))
documents_2=south_asia

east_and_south_east_asia=[]
for i in data_3.Headline.values:
    east_and_south_east_asia.append(str(i))
documents_3=east_and_south_east_asia

europe_and_central_asia=[]
for i in data_4.Headline.values:
    europe_and_central_asia.append(str(i))
documents_4=europe_and_central_asia

central_africa=[]
for i in data_5.Headline.values:
    central_africa.append(str(i))
documents_5=central_africa

east_africa=[]
for i in data_6.Headline.values:
    east_africa.append(str(i))
documents_6=east_africa

southern_africa=[]
for i in data_7.Headline.values:
    southern_africa.append(str(i))
documents_7=southern_africa

middle_east_and_north_africa=[]
for i in data_8.Headline.values:
    middle_east_and_north_africa.append(str(i))
documents_8=middle_east_and_north_africa

north_america=[]
for i in data_9.Headline.values:
    north_america.append(str(i))
documents_9=north_america

latin_america_and_caribbean=[]
for i in data_10.Headline.values:
    latin_america_and_caribbean.append(str(i))
documents_10=latin_america_and_caribbean

west_africa=[]
for i in data_11.Headline.values:
    west_africa.append(str(i))
documents_11=west_africa

train_set_0 = pd.Series(documents_0)
train_set_1 = pd.Series(documents_1)
train_set_2 = pd.Series(documents_2)
train_set_3 = pd.Series(documents_3)
train_set_4 = pd.Series(documents_4)
train_set_5 = pd.Series(documents_5)
train_set_6 = pd.Series(documents_6)
train_set_7 = pd.Series(documents_7)
train_set_8 = pd.Series(documents_8)
train_set_9 = pd.Series(documents_9)
train_set_10 = pd.Series(documents_10)
train_set_11 = pd.Series(documents_11)

#training dataset
tokenizer = TfidfVectorizer()
tokenizer.fit(train_set_0)
tokenizer.fit(train_set_1)
tokenizer.fit(train_set_2)
tokenizer.fit(train_set_3)
tokenizer.fit(train_set_4)
tokenizer.fit(train_set_5)
tokenizer.fit(train_set_6)
tokenizer.fit(train_set_7)
tokenizer.fit(train_set_8)
tokenizer.fit(train_set_9)
tokenizer.fit(train_set_10)
tokenizer.fit(train_set_11)

train_set_00 = tokenizer.transform(train_set_0)
train_set_01 = tokenizer.transform(train_set_1)
train_set_02 = tokenizer.transform(train_set_2)
train_set_03 = tokenizer.transform(train_set_3)
train_set_04 = tokenizer.transform(train_set_4)
train_set_05 = tokenizer.transform(train_set_5)
train_set_06 = tokenizer.transform(train_set_6)
train_set_07 = tokenizer.transform(train_set_7)
train_set_08 = tokenizer.transform(train_set_8)
train_set_09 = tokenizer.transform(train_set_9)
train_set_010 = tokenizer.transform(train_set_10)
train_set_011 = tokenizer.transform(train_set_11)


#creating pickle files
with open('./pickle_files/train_set_0.pickle', 'wb') as a_handle:
    pickle.dump(train_set_00, a_handle)

with open('./pickle_files/train_set_1.pickle', 'wb') as b_handle:
    pickle.dump(train_set_01, b_handle)

with open('./pickle_files/train_set_2.pickle', 'wb') as c_handle:
    pickle.dump(train_set_02, c_handle)

with open('./pickle_files/train_set_3.pickle', 'wb') as d_handle:
    pickle.dump(train_set_03, d_handle)

with open('./pickle_files/train_set_4.pickle', 'wb') as e_handle:
    pickle.dump(train_set_04, e_handle)

with open('./pickle_files/train_set_5.pickle', 'wb') as f_handle:
    pickle.dump(train_set_05, f_handle)

with open('./pickle_files/train_set_6.pickle', 'wb') as g_handle:
    pickle.dump(train_set_06, g_handle)

with open('./pickle_files/train_set_7.pickle', 'wb') as h_handle:
    pickle.dump(train_set_07, h_handle)

with open('./pickle_files/train_set_8.pickle', 'wb') as i_handle:
    pickle.dump(train_set_08, i_handle)

with open('./pickle_files/train_set_9.pickle', 'wb') as j_handle:
    pickle.dump(train_set_09, j_handle)

with open('./pickle_files/train_set_10.pickle', 'wb') as k_handle:
    pickle.dump(train_set_010, k_handle)

with open('./pickle_files/train_set_11.pickle', 'wb') as l_handle:
    pickle.dump(train_set_011, l_handle)

filename = './pickle_files/finalized_model.sav'
joblib.dump(tokenizer, filename)
