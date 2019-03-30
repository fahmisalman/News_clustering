from sklearn.cluster import KMeans
from sklearn import metrics
from Preprocessing import *
import glob
import TFIDF
import csv


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


if __name__ == '__main__':

    data = load_data()
    data.insert(0, 'content')
    with open('Model_data/Article.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([data])
    # temp = load_data()
    #
    # for i in range(len(data)):
    #     data[i] = preprocessing(data[i])
    #     data[i] = list(filter(None, data[i]))
    #
    # tfidf, words = TFIDF.termfreq(data)
    #
    # a = [2, 3, 5, 10]
    #
    # for i in a:
    #     model = KMeans(n_clusters=i, random_state=1).fit(tfidf)
    #     print('{}\t{}\t{}'.format(i, metrics.silhouette_score(tfidf, model.labels_, metric='euclidean'),
    #                               metrics.davies_bouldin_score(tfidf, model.labels_)
    #                               ))
    #
    #     with open('Model_data/label_cluster-{}.csv'.format(i), 'w') as f:
    #         writer = csv.writer(f)
    #         writer.writerows([model.labels_])
