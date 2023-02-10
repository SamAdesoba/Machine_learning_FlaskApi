import flask
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

model = pickle.load(open('model/model_pickle.pkl', 'rb'))

vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), max_df=500)

df = pd.read_csv('util/atiku.csv')

df_tweet = df['tweet']


def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9^\w]+', '', text)
    text = re.sub(r'#[A-Za-z0-9^\w]+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
    text = re.sub(r'\:', '', text)
    text = re.sub(r'\...', '', text)
    return text


def reformat_json(text):
    text = re.sub(r'\"', ' ', text)
    text = re.sub(r'\,.', ' ', text)
    text = re.sub(r'\(', ' ', text)
    text = re.sub(r'\)..', ' ', text)
    return text


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def sentiment():
    cleaned_data = df_tweet.apply(cleanText)

    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])

    vectorizer.fit(clean_df['tweet'].values)

    vectorized = vectorizer.transform(clean_df['tweet'])

    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())

    result = model.predict(vectorized_df.values)

    result_df = pd.DataFrame(result, columns=['Analysis'])

    format_result = result_df.value_counts().to_json(orient='columns')

    return reformat_json(format_result)
