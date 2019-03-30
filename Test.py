from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer
from Preprocessing import *

import nltk
import glob

from sklearn import cluster
from sklearn import metrics

# training data


def load_data():
    d = []
    for name in glob.glob('Dataset/*'):
        temp = open(name)
        d.append(temp.read())
    return d


def preprocessing(sentence):
    sentence = caseFolding(sentence)
    token = tokenization(sentence)
    token = stopwordRemoval(token)
    for i in range(len(token)):
        token[i] = stemming(token[i])
    return token


data = load_data()

for i in range(len(data)):
    data[i] = preprocessing(data[i])

# sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
#              ['this', 'is', 'another', 'book'],
#              ['one', 'more', 'book'],
#              ['this', 'is', 'the', 'new', 'post'],
#              ['this', 'is', 'about', 'machine', 'learning', 'post'],
#              ['and', 'this', 'is', 'the', 'last', 'post']]

# training model
model = Word2Vec(data, min_count=1)

# get vector data
X = model[model.wv.vocab]
print(X)

# print(model.similarity('this', 'is'))
#
# print(model.similarity('post', 'book'))
#
# print(model.most_similar(positive=['machine'], negative=[], topn=2))
#
# print(model['the'])

print(list(model.wv.vocab))

print(len(list(model.wv.vocab)))

NUM_CLUSTERS = 3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print(assigned_clusters)

words = list(model.wv.vocab)
for i, word in enumerate(words):
    print(word + ":" + str(assigned_clusters[i]))

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)

print(
    "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)