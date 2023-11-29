import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

try:
    df = pd.read_csv("books.csv", on_bad_lines="skip")
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

df.index = df["bookID"]

# Finding Number of rows and columns
print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))
df.head()
df.replace(to_replace="J.K. Rowling/Mary GrandPrÃ©", value="J.K. Rowling", inplace=True)
df.head()


df.average_rating.isnull().value_counts()

trial = df[["average_rating", "ratings_count"]]
data = np.asarray(
    [np.asarray(trial["average_rating"]), np.asarray(trial["ratings_count"])]
).T

X = data
distortions = []
for k in range(2, 30):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distortions.append(k_means.inertia_)

# Computing K means with K = 5, thus, taking it as 5 clusters
centroids, _ = kmeans(data, 5)

idx, _ = vq(data, centroids)


trial.idxmax()


trial = trial[~trial.index.isin([3, 41865])]

data = np.asarray(
    [np.asarray(trial["average_rating"]), np.asarray(trial["ratings_count"])]
).T

centroids, _ = kmeans(data, 5)

idx, _ = vq(data, centroids)

books_features = pd.concat([df["average_rating"], df["ratings_count"]], axis=1)

books_features.head()

min_max_scaler = MinMaxScaler()
books_features = min_max_scaler.fit_transform(books_features)

np.round(books_features, 2)

model = neighbors.NearestNeighbors(n_neighbors=6, algorithm="ball_tree")
model.fit(books_features)
distance, indices = model.kneighbors(books_features)


def get_index_from_name(name):
    return df[df["title"] == name].index.tolist()[0]


def get_id_from_partial_name(partial):
    all_books_names = list(df.title.values)
    l = []
    for name in all_books_names:
        if partial in name:
            l.append((name, all_books_names.index(name)))
    return l


def find_similar_books_by_author(authors):
    similar_books = df[df["authors"] == authors]["title"].tolist()
    return similar_books


def find_similar_books_by_publisher(publisher):
    similar_books = df[df["publisher"] == publisher]["title"].tolist()
    return similar_books


def print_similar_books(query=None, id=None, authors=None, publisher=None):
    if id:
        for id in indices[id][1:]:
            print(df.iloc[id]["title"])
    if query:
        found_id = get_index_from_name(query)
        for id in indices[found_id][1:]:
            print(df.iloc[id]["title"])
    if authors:
        similar_books = find_similar_books_by_author(authors)
        for book_title in similar_books:
            print(book_title)
    if publisher:
        similar_books = find_similar_books_by_publisher(publisher)
        for book_title in similar_books:
            print(book_title)


def recommend_books_by_average_rating(num_recommendations=5):
    top_rated_books = df.sort_values(by="average_rating", ascending=False)[
        ["title", "average_rating"]
    ].head(num_recommendations)

    top_rated_books["combined"] = (
        top_rated_books["title"]
        + " - ["
        + top_rated_books["average_rating"].astype(str)
        + "]"
    )
    combined_books = top_rated_books["combined"].tolist()
    return combined_books
