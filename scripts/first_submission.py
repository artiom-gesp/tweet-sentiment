import pandas as pd
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

df_train_pos = pd.read_table('../data/train_pos.txt', header=None, names=['tweet'])
df_train_neg = pd.read_table('../data/train_neg.txt', header=None, names=['tweet'])
df_test = pd.read_table('../data/test_data.txt', header=None, names=['tweet'])

df_train_pos['sentiment'] = 1
df_train_neg['sentiment'] = 0

df_train = pd.concat([df_train_pos, df_train_neg])

# spacy pipeline
# English pipeline optimized for CPU. 
# Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.
# https://spacy.io/models/en
nlp = spacy.load('en_core_web_lg')

# punctuation and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

def tweet_cleaner(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    clean_tokens = []
    for token in tokens:
        if (token not in punctuations) and (token not in stop_words):
            clean_tokens.append(token)
    return clean_tokens

# custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        """Override the transform method to clean text"""
        collector = []
        for text in tqdm(X, total=len(X), desc='Cleaning text:\t'):
            collector.append(clean_text(text))
        return collector
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_params(self, deep=True):
        return {}

# basic function to clean the text
def clean_text(text):
    """Removing spaces and converting the text into lowercase"""
    return text.strip().lower()


# different vectorizers
bow_vector = CountVectorizer(tokenizer=tweet_cleaner, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer=tweet_cleaner)
hash_vector = HashingVectorizer(tokenizer=tweet_cleaner)

X = df_train['tweet']
y = df_train['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 42)
print(f'X_train dimension: {X_train.shape}')
print(f'y_train dimension: {y_train.shape}')
print(f'X_test dimension: {X_test.shape}')
print(f'y_train dimension: {y_test.shape}')


classifier = LogisticRegression(verbose=1, solver='lbfgs', max_iter=10000)
# classifier = MLPClassifier(hidden_layer_sizes=(256,128,64), verbose=True)


# Create pipeline using Bag of Words
components = [
    ("cleaner", predictors()),
    ("vectorizer", hash_vector),
    ("classifier", classifier)
        ]
pipe = Pipeline(components)

# Test with 1/100 of the data to estimate the time needed
before = time.time()
pipe.fit(X_train[:len(X_train)//1000], y_train[:len(y_train)//1000])
after = time.time()
print(f'\n\nTime needed for a 1000th ({len(X_train)//1000} samples): {after-before} s')
print(f'Time needed for the whole dataset ({len(X_train)} samples): {(after-before)*1000} s\n\n')

# Model generation
# pipe.fit(X_train[:len(X_train)//1000], y_train[:len(y_train)//1000])
pipe.fit(X_train, y_train)


from sklearn import metrics
# Model accuracy score
predicted = pipe.predict(X_test)
print(f'Accuracy: {metrics.accuracy_score(y_test, predicted)}')
print(f'Precision: {metrics.precision_score(y_test, predicted)}')
print(f'Recall: {metrics.recall_score(y_test, predicted)}')

# Predicting with test dataset
predicted = pipe.predict(df_test['tweet'])
submission = pd.DataFrame(predicted, columns=['prediction'])
submission.index += 1
submission.to_csv('../submission/first_submission_lg_ha.csv', index_label='Id')
print('Saved result.') 
