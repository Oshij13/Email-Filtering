import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv(r"C:\Users\oshij\OneDrive\Pictures\Desktop\spam_ham_dataset.csv")
print(df.head(2))
df.info()
df.describe()

print(df.groupby("label").describe())

df["Length"] = df["text"].apply(len)
print(df.head())

sns.displot(df["Length"], bins=30)
plt.show()

print(df["Length"].max())

print(df[df["Length"]==32258]["text"].iloc[0])

stop_words = set(stopwords.words('english'))


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

print(df['text'].head().apply(process_text))

#message_bow = CountVectorizer(analyzer= process_text).fit_transform(df['text'])
#print(message_bow)

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(message_bow, df['label'], test_size=20, random_state=0)

#from sklearn.naive_bayes import MultinomialNB
#classifier = MultinomialNB().fit(X_train,Y_train)

#print(classifier.predict(X_train))
#print(Y_train.values)

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#pred = classifier.predict(X_train)
#print(classification_report(Y_train, pred))

#print(classifier.predict(X_test))
#print(Y_train.values)