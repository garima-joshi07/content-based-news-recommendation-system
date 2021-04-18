from flask import Flask,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd,pickle, re
#import joblib
#nltk.download('stopwords') # load english stopwords
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer#added
import re#added
import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify
import json
import html2text
import pickle
import joblib
from time import time

#unpickle file
with open('./pickle_files/train_set_0.pickle', 'rb') as a_handle:
    set_0 = pickle.load(a_handle)

with open('./pickle_files/train_set_1.pickle', 'rb') as b_handle:
    set_1 = pickle.load(b_handle)

with open('./pickle_files/train_set_2.pickle', 'rb') as c_handle:
    set_2 = pickle.load(c_handle)

with open('./pickle_files/train_set_3.pickle', 'rb') as d_handle:
    set_3 = pickle.load(d_handle)

with open('./pickle_files/train_set_4.pickle', 'rb') as e_handle:
    set_4 = pickle.load(e_handle)

with open('./pickle_files/train_set_5.pickle', 'rb') as f_handle:
    set_5 = pickle.load(f_handle)

with open('./pickle_files/train_set_6.pickle', 'rb') as g_handle:
    set_6 = pickle.load(g_handle)

with open('./pickle_files/train_set_7.pickle', 'rb') as h_handle:
    set_7 = pickle.load(h_handle)

with open('./pickle_files/train_set_8.pickle', 'rb') as i_handle:
    set_8 = pickle.load(i_handle)

with open('./pickle_files/train_set_9.pickle', 'rb') as j_handle:
    set_9 = pickle.load(j_handle)

with open('./pickle_files/train_set_10.pickle', 'rb') as j_handle:
    set_10 = pickle.load(j_handle)

with open('./pickle_files/train_set_11.pickle', 'rb') as k_handle:
    set_11 = pickle.load(k_handle)
    
  
global newlist
newlist=[set_0, set_1, set_2, set_3, set_4, set_5, set_6, set_7, set_8, set_9, set_10, set_11]
    

app = Flask(__name__)
filename = './pickle_files/finalized_model.sav'
loaded_model = joblib.load(filename)


@app.route('/', methods=['GET'])
def Index():
    return "Hello"

@app.route('/prediction', methods=['POST'])
def prediction():
    html = request.form['content']
    listContent = []
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text = text_maker.handle(html)
    listContent.append(text)
    tuple_data = tuple(listContent)
    test= pd.DataFrame(tuple_data, columns=["article_headline"])
    set=[]
    for i in test.article_headline.values:
        set.append(str(i))
    document =set
    test_set = pd.Series(document)
    new_tfidf = loaded_model.transform(test_set)
    Editions= ['Global_Edition','Pacific', 'South_Asia', 'East_and_South_East_Asia','Europe_and_Central_Asia','Central_Africa','East_Africa','Southern_Africa','Middle_East_and_North_Africa','North_America','Latin_America_and_Caribbean','West_Africa']
    EditionIds = ['FF42A092-3B04-4883-A92B-B10520A2DAD9','6C0D01D2-37AB-44F9-993F-EEEDC2EB5917', '8AE4B6A3-87C2-45BA-B748-B81A1FBBC2FA', '8EF70674-7DBA-432A-A8A4-FCE4C03349D0', 'B5D6873F-E7D9-43F1-B81C-5CF85FEBFFC4', 'E8A300D3-F7EF-42CB-A35E-39686FEA2A6D', '748A1A40-3ADE-4DDE-ADC2-3316EF0563AD', '68D8B03E-6681-4513-9337-996E201BC148', '5CAA1C80-E148-4AA3-B0DB-3115A2965691', '42C9EE0D-BDFA-4AC0-A2A4-EF69392BDBF5', '7F42122F-755D-4548-B8DE-EEF5F9DEC952', '1478FD49-5D51-443B-9161-8A57E8FDD7F1']
    score_value=[]
    l2=[]
    for i in range(len(newlist)):
        #new_tfidf = tokenizer.transform(new_series)
        a = cosine_similarity(newlist[i], new_tfidf).mean()
        dictionary={}
        dictionary['region']=Editions[i]
        dictionary['id']=EditionIds [i]
        dictionary['score']=a.tolist() 
        l2.append(dictionary)
    print(score_value)
    json_object = json.dumps(sorted(l2,key=lambda x: x['score']))
    toc = time()  
    print("[INFO] Total time taken: ", round((toc - tic), 2), " seconds.")
    return(json_object)


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=2009)