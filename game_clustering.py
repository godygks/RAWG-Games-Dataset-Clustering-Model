import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_sample_data(filepath, sample_size=50000):
    df = pd.read_csv(filepath)
    features = ['genres', 'tags', 'platforms', 'developers']
    df = df[features].dropna()
    return df.sample(n=sample_size, random_state=42)

def count_developers(df):
    df['num_developers'] = df['developers'].apply(lambda x: len(str(x).split(',')))
    return df.drop(columns=['developers'])

def filter_top_n_multilabel(series, top_n=30):
    all_items = [item.strip() for sublist in series.dropna().apply(lambda x: str(x).split(',')) for item in sublist]
    common = set([label for label, _ in Counter(all_items).most_common(top_n)])
    return series.apply(lambda x: [item.strip() for item in str(x).split(',') if item.strip() in common])

def one_hot_encode_multilabel(df, column, top_n=50):
    df[column] = df[column].fillna('Unknown')
    df[column] = filter_top_n_multilabel(df[column], top_n=top_n)
    mlb = MultiLabelBinarizer()
    encoded = pd.DataFrame(mlb.fit_transform(df[column]), columns=[f"{column}_{cls}" for cls in mlb.classes_])
    encoded.index = df.index
    return encoded

def preprocess_features(df):
    encoded_parts = []
    for col in ['genres', 'tags', 'platforms']:
        encoded_parts.append(one_hot_encode_multilabel(df, col, top_n=50))
    encoded_df = pd.concat(encoded_parts, axis=1)
    encoded_df['num_developers'] = df['num_developers']
    encoded_df = pd.get_dummies(encoded_df, columns=['num_developers'], prefix='num_devs')
    return encoded_df

def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def run_kmeans_and_plot(X, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='Set2', legend='full')
    plt.title(f"PCA 2D Visualization with KMeans (k={k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()
    return labels

def main():
    filepath = "rawg_games_data.csv"  # <-- Replace with your dataset path
    df = load_and_sample_data(filepath)
    df = count_developers(df)
    X = preprocess_features(df)
    X_pca = run_pca(X, n_components=2)
    run_kmeans_and_plot(X_pca, k=5)

if __name__ == "__main__":
    main()
