#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pickle
import pandas as pd
from tokenize import tokenize
from sklearn.metrics import recall_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from textblob import TextBlob
import re
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# # Retrieving Dataset

# In[2]:


master_tweets = pd.read_csv('Data\\11th_hour_political_tweets_formatted.csv')
# dt=pd.read_csv('Dictionary\\Master_Dictionary.csv')


# # DataProcessing

# In[3]:


contractions = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have",
    "aint": "am not / are not",
    "arent": "are not / am not",
    "cant": "cannot",
    "cause": "because",
    "couldve": "could have",
    "couldnt": "could not",
    "didnt": "did not",
    "doesnt": "does not",
    "dont": "do not",
    "hadnt": "had not",
    "hasnt": "has not",
    "havent": "have not",
    "hed": "he had / he would",
    "hell": "he shall / he will",
    "hes": "he has / he is",
    "howd": "how did",
    "howll": "how will",
    "hows": "how has / how is",
    "id": "I had / I would",
    "ill": "I shall / I will",
    "im": "I am",
    "ive": "I have",
    "isnt": "is not",
    "itd": "it had / it would",
    "itll": "it shall / it will",
    "its": "it has / it is",
    "lets": "let us",
    "maam": "madam",
    "maynt": "may not",
    "mightve": "might have",
    "mightnt": "might not",
    "mustve": "must have",
    "mustnt": "must not",
    "neednt": "need not",
    "oclock": "of the clock",
    "oughtnt": "ought not",
    "shant": "shall not",
    "shantve": "shall not have",
    "shed": "she had / she would",
    "shell": "she shall / she will",
    "shes": "she has / she is",
    "shouldve": "should have",
    "shouldnt": "should not",
    "sove": "so have",
    "sos": "so as / so is",
    "thatd": "that would / that had",
    "thats": "that has / that is",
    "thered": "there had / there would",
    "theres": "there has / there is",
    "theyd": "they had / they would",
    "theyll": "they shall / they will",
    "theyre": "they are",
    "theyve": "they have",
    "tove": "to have",
    "wasnt": "was not",
    "wed": "we had / we would",
    "well": "we will",
    "were": "we are",
    "weve": "we have",
    "werent": "were not",
    "whatll": "what shall / what will",
    "whatre": "what are",
    "whats": "what has / what is",
    "whatve": "what have",
    "whens": "when has / when is",
    "whenve": "when have",
    "whered": "where did",
    "wheres": "where has / where is",
    "whereve": "where have",
    "wholl": "who shall / who will",
    "whos": "who has / who is",
    "whove": "who have",
    "whys": "why has / why is",
    "whyve": "why have",
    "willve": "will have",
    "wont": "will not",
    "wontve": "will not have",
    "wouldve": "would have",
    "wouldnt": "would not",
    "yall": "you all",
    "youd": "you had / you would",
    "youll": "you shall / you will",
    "youre": "you are",
    "youve": "you have",
    "narendramodi": "Good",
    "tame": "you",
    "namaste": "salute",
    "aap": "you",
    "aapki": "yours",
    "PB": "punjab",
    "Har": "hariyana",
    "HP": "himachal pradesh",
    "UP": "uattar pradesh",
    "UK": "uattrakhand",
    "Guj": "gujarat",
    "Raj": "rajasthan",
    "MP": "madhyapradesh",
    "CG": "chattisgadh",
    "Bih": "bihar",
    "MH": "maharastra",
    "Kar": "karantaka",
    "TN": "tamilnadu",
    "AP": "andrapradesh",
    "TG": "telengana",
    "Ker": "kerela",
    "WB": "westbengal",
    "Ori": "orissa",
    "UTs": "union teretories",
    "*without legal counsel* is a *statement*.": "without legal counsel is a statement",
    "NE": "north east",
    "IndiaBoleModiDobara / NaMoAgain": " good again",
    "Thatﾒs": "thats",
    "Aapka kuch nahi ho sakta bhai": "Nothing good can be done to you",
    "Phir": "again",
    "nahi Deta hai": "not given",
    "gali": "a small street",
    "suppt": "support",
    "Govt": "goverment",
    "MainBhiBerozgar": "Me too Unemployed",
    "Samjho thugs Kuch Samjho": "Learn something deceivers learn",
    "MainBhiChorHoon": "I am also a thief",
    "Farzi": "Fake",
    "ghatia": "Rubbish",
    "Dalits": "low caste people",
    "sharam": "shame",
    "nhi": "not",
    "hota": "happen",
    "tab": "then",
    "chize": "things",
    "ki": "of",
    "jehadi": "an Islamic militant.",
    "mara": "killed",
    "desh": "country",
    "gussa": "anger",
    "ayega": "will come",
    "MARD": "MAN",
    "Kya": "what",
    "ap": "you",
    "mujhse": "with me",
    "mil": "meet",
    "sakte": "can",
    "dekhti": "Watching",
    "Aap": "you",
    "garib": "poor",
    "bachi": "baby girl",
    "kaise": "how",
    "karenge": "will do",
    "aaraam": "Rest",
    "Aapko": "you",
    "milega": "get",
    "Pappu": "Mad child",
    "Kaamdhari": "worker",
    "Fauj": "army",
    "Bharat": "India",
    "Musalman": "Muslim",
    "hoon": "i am",
    "dharm": "religion",
    "poojne": "worship",
    "Mera": "my",
    "Parivar": "Family",
    "Saath": "together",
    "Chalte": "walk",
    "Chale Jao": "walk away",
    "chacha": "uncle",
    "ppl": "people",
    "it?s": "its",
    "2h2": "2hours",
    "Hon'ble": "honorable",
    "r": "are",
    "u": "you",
    "C'mon": "common",
    "PS": "post-scriptum",
    "wiki": "wikipedia",
    "ain't": "are not/is not/ have not/has not",
    "*is*": "is",
    "aam": "normal",
    "Chor Hai": "robber",
    "didi": "sister",
    "1st": "first",
    "Lok Sabha": "meeting",
    "CM": "chief minister"

}


# In[4]:


# Hashtags
hash_regex = re.compile(r"#(\w+)")


def hash_repl(match):
    return ''+match.group(1).upper()


# Handels
hndl_regex = re.compile(r"@(\w+)")


def hndl_repl(match):
    return ''


# URLs
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

# Spliting by word boundaries
word_bound_regex = re.compile(r"\W+")

# Repeating words like hurrrryyyyyy
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)


def rpt_repl(match):
    return match.group(1)+match.group(1)


# Emoticons
emoticons = [('__EMOT_SMILEY', [':-)', ':)', '(:', '(-:', ]),  ('__EMOT_LAUGH',  [':-D', ':D', 'X-D', 'XD', 'xD', ]),  ('__EMOT_LOVE',  ['<3', ':\*', ]),
             ('__EMOT_WINK',  [';-)', ';)', ';-D', ';D', '(;', '(-;', ]),  ('__EMOT_FROWN',  [':-(', ':(', '(:', '(-:', ]),  ('__EMOT_CRY',  [':,(', ':\'(', ':"(', ':((']), ]

# Punctuations
punctuations = [
    ('',		['!', '¡', ])	,
    ('',		['?', '¿', ])	,
    ('',		['...', '…', ])	,\
    # FIXME : MORE? http://en.wikipedia.org/wiki/Punctuation
]

# Printing functions for info


def print_config(cfg):
    for (x, arr) in cfg:
        print(x, '\t')
        for a in arr:
            print(a, '\t')
        print('')


def print_emoticons():
    print_config(emoticons)


def print_punctuations():
    print_config(punctuations)

# For emoticon regexes


def escape_paren(arr):
    return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]


def regex_union(arr):
    return '(' + '|'.join(arr) + ')'


emoticons_regex = [(repl, re.compile(regex_union(escape_paren(regx))))
                   for (repl, regx) in emoticons]

# For punctuation replacement


def punctuations_repl(match):
    text = match.group(0)
    repl = []
    for (key, parr) in punctuations:
        for punc in parr:
            if punc in text:
                repl.append(key)
    if(len(repl) > 0):
        return ' '+' '.join(repl)+' '
    else:
        return ' '


def processHashtags(text, subject='', query=[]):
    return re.sub(hash_regex, hash_repl, text)


def processHandles(text, subject='', query=[]):
    return re.sub(hndl_regex, hndl_repl, text)


def processUrls(text, subject='', query=[]):
    return re.sub(url_regex, '', text)


def processEmoticons(text, subject='', query=[]):
    for (repl, regx) in emoticons_regex:
        text = re.sub(regx, ' '+repl+' ', text)
    return text


def processPunctuations(text, subject='', query=[]):
    return re.sub(word_bound_regex, punctuations_repl, text)


def processRepeatings(text, subject='', query=[]):
    return re.sub(rpt_regex, rpt_repl, text)


def processQueryTerm(text, subject='', query=[]):
    query_regex = "|".join([re.escape(q) for q in query])
    return re.sub(query_regex, '', text, flags=re.IGNORECASE)


def countHandles(text):
    return len(re.findall(hndl_regex, text))


def countHashtags(text):
    return len(re.findall(hash_regex, text))


def countUrls(text):
    return len(re.findall(url_regex, text))


def countEmoticons(text):
    count = 0
    for (repl, regx) in emoticons_regex:
        count += len(re.findall(regx, text))
    return count


def clean_sen(text):
    for word in text.split():
        if word in contractions:
            text = text.replace(word, contractions[word])
    return text.lower()

# FIXME: use process functions inside


def processAll(text, subject='', query=[]):

    if(len(query) > 0):
        query_regex = "|".join([re.escape(q) for q in query])
        text = re.sub(query_regex, '', text, flags=re.IGNORECASE)

    text = re.sub(hash_regex, hash_repl, text)
    text = re.sub(hndl_regex, hndl_repl, text)
    text = re.sub(url_regex, '', text)

    for (repl, regx) in emoticons_regex:
        text = re.sub(regx, ' '+repl+' ', text)

    text = text.replace('\'', '')

    text = re.sub(word_bound_regex, punctuations_repl, text)
    text = re.sub(rpt_regex, rpt_repl, text)

    return clean_sen(text)


# In[5]:


master_tweets['tweet_list'] = master_tweets['tweet'].apply(processAll)


# In[6]:


master_tweets.iloc[np.random.permutation(len(master_tweets))]


# In[7]:


# Classifying the good and bad dataframes
good = master_tweets[master_tweets['label'] == 0]
bad = master_tweets[master_tweets['label'] == 1]


# In[8]:


master_tweets = pd.concat([good[:18000], bad[:18000]])


# # Fitting Dataset into Train Test

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(
    master_tweets['tweet_list'], master_tweets['label'], test_size=0.2, random_state=np.random)


# # TF-IDF

# In[13]:


tfidf = TfidfVectorizer(smooth_idf=True, use_idf=True)
X_train_dtm = tfidf.fit_transform(X_train)
X_test_dtm = tfidf.transform(X_test)


# # Logistic regression

# In[25]:


LR = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear',
                        C=3, fit_intercept=True, intercept_scaling=3.0, warm_start=True)
LR.fit(X_train_dtm, y_train)
y_pred = LR.predict(X_test_dtm)
print('\nLogistic Regression')
print('Accuracy Score: ', metrics.accuracy_score(
    y_test, y_pred)*100, '%', sep='')
print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')
print(classification_report(y_pred, y_test))


# In[27]:


pickle.dump(LR, open('Portable_Model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))


# In[ ]:


# In[ ]:
