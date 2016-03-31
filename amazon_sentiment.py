# -*-coding:utf-8 -*-

'''By YinHao
   Text classfier
'''
import sqlite3
import pandas as pd
import numpy as np
import nltk
#nltk.download()
import string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from nltk.stem.porter import PorterStemmer

sql_connect = sqlite3.connect('C:/Users/b51454/Desktop/task/amazon-fine-foods/database.sqlite')

messages = pd.read_sql_query("""
    SELECT Score,Summary
    FROM Reviews
    WHERE Score != 3
    """,sql_connect)

def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'

Score = messages['Score']
Score = Score.map(partition)
Summary = messages['Summary']

X_train,X_test,y_train,y_test = train_test_split(Summary,Score,test_size = 0.2,random_state = 42)

print("hello")
print(len(X_train))
print("hello")
print(len(X_test))
#print()
#print(messages.tail(20))

tmp = messages
tmp['Score'] = tmp['Score'].map(partition)
#print(tmp.head(20))

stemmer = PorterStemmer()
from nltk.corpus import stopwords

#英文分词
#中文为jieba分词
from nltk.tokenize import word_tokenize
def stem_tokens(tokens,stemmer):
    stemmed =[]
    for item in tokens:
        word = stemmer.stem(item)
        #if word not in stopwords.words('english'):
        stemmed.append(word)
    return stemmed

#分割句子
def tokenize(text):
    tokens = word_tokenize(text)  
    stems = stem_tokens(tokens,stemmer)
    return ' '.join(stems)

#去掉标点符号
intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)

#对训练词矩阵进行预处理
corpus = []
for text in X_train:
    text = text.lower()
    text = text.translate(trantab)
    text = tokenize(text)
    corpus.append(text)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("TfidfTransformer")
print((X_train_tfidf[0].shape))

#对测试词矩阵进行预处理
test_set = []
for text in X_test:
    text = text.lower()
    text = text.translate(trantab)
    text=tokenize(text)
    test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)
#print("X_test_tfidf")
#print(X_test_tfidf[0].shape)

from pandas import *
df = DataFrame({'Before':X_train,'After':corpus})
print(df.head(20))

prediction = dict()

#从sklearn中选择二项bayes,逻辑回归，不带核的SVM(带核的速度太慢）进行训练
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train_tfidf,y_train)
prediction['BernoulliNB'] = model.predict(X_test_tfidf)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_tfidf,y_train)
prediction['logistic'] = logreg.predict(X_test_tfidf)

from sklearn import svm
model = svm.SVC(kernel = 'linear').fit(X_train_tfidf,y_train)
prediction['SVM_linear'] = model.predict(X_test_tfidf)

#model = svm.SVC(kernel = 'rbf').fit(X_train_tfidf,y_train)
#prediction['SVM_rbf'] = model.predict(X_test_tfidf)
def formatt(x):
    if x == 'negative':
        return 0
    return 1

vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b','g','y','m','k']

#选择AUC和ROC
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title("classifiers comparaison with ROC")
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Postitive Rate')
plt.xlabel('False Postitive Rate')
plt.show()

#经过参数调优后，发现LR和SVM的AUC较高(两者相差不大）
print("logistic results")
print(metrics.classification_report(y_test,prediction['logistic'],target_names = ["positive","negative"]))

print("SVM_linear results")
print(metrics.classification_report(y_test,prediction['SVM_linear'],target_names = ["positive","negative"]))

#print(metrics.classification_report(y_test,prediction['logistic'],target_names = ["positive","negative"]))``