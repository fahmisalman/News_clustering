from Preprocessing_2 import *
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import matplotlib.pyplot as plt
import pickle
import glob


def load_data(loc):

    d = []
    for filename in glob.glob(loc):
        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            for data in temp:
                d.append(data)
    return d


def preprocessing(d):
    d = d.lower()
    d = remove_escape(d)
    d = remove_url(d)
    d = remove_punctuation(d)
    d = stopword_removal(d)
    return d


if __name__ == '__main__':

    raw = load_data('pkl/*')
    news = []
    for row in raw:
        news.append(row['news_title'])

    mask = 'Assets/Mask/twitter_mask.png'
    font = 'Assets/Font/CabinSketch-Bold.ttf'
    background = 'white'

    for i in range(len(news)):
        news[i] = preprocessing(news[i])
    words = ' '.join(news)

    # twitter_mask = imread(mask, flatten=True)
    wordcloud = WordCloud(
        stopwords=STOPWORDS,
        font_path=font,
        background_color=background
        # width=twitter_mask.shape[1],
        # height=twitter_mask.shape[0],
        # mask=twitter_mask
    ).generate(words)

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
