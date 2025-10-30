"""extract_keywords.py

Small command-line script to load a CSV with 'review' and 'sentiment' columns,
and produce a CSV with ranked negative-review keywords using TF-IDF.
"""
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import numpy as np

def main(input_csv, output_csv, top_n=50):
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    df = pd.read_csv(input_csv)
    neg_df = df[df['sentiment']=='negative'].copy()
    corpus = neg_df['review'].astype(str).tolist()
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words=stop_words, max_df=0.85, min_df=1)
    X = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    avg_tfidf = np.asarray(X.mean(axis=0)).ravel()
    df_counts = ((X.toarray())>0).sum(axis=0)
    scores = df_counts * avg_tfidf
    out = pd.DataFrame({
        'phrase': terms,
        'doc_freq': df_counts,
        'avg_tfidf': avg_tfidf,
        'score': scores
    })
    out = out.sort_values('score', ascending=False).head(top_n)
    out.to_csv(output_csv, index=False)
    print(f'Saved top {top_n} negative keywords to {output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='reviews.csv')
    parser.add_argument('--output', default='negative_keywords.csv')
    parser.add_argument('--top_n', type=int, default=50)
    args = parser.parse_args()
    main(args.input, args.output, args.top_n)
