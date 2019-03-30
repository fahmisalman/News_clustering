from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

factory = StemmerFactory()
stemmer = factory.create_stemmer()
# rootwords = [line.rstrip('\n\r') for line in open('rootwords.txt')]


def caseFolding(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'''[']''', '', sentence)
    sentence = re.sub(r'[^a-z]', ' ', sentence)
    return sentence


def tokenization(sentence):
    return sentence.split()


def stemming(token):
    stem = stemmer.stem(token)
    # if stem in rootwords:
    #     return stem
    return stem


def stopwordRemoval(token):
    stopword = [line.rstrip('\n\r') for line in open('stopwords.txt')]
    temp = []
    for i in range(len(token)):
        if token[i] not in stopword:
            temp.append(token[i])
    return temp
