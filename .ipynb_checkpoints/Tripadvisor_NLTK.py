
data = pd.read_csv(data_file)
rating = pd.to_numeric(data['review_rating'])
positive_reviews = data['review_title'][rating > 3]
negative_reviews = data['review_title'][rating <= 3]
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.tag import pos_tag

def tokenise_words(positive_reviews, negative_reviews):
    positive_tokens = list()
    negative_tokens = list()
    for pos_sentence  in positive_reviews:
        add_pos_sentence = word_tokenize(pos_sentence)
        positive_tokens.append(add_pos_sentence)

    for neg_sentence in negative_reviews:
        add_neg_sentence = word_tokenize(neg_sentence)
        negative_tokens.append(add_neg_sentence)
    return zip(positive_tokens, negative_tokens)

def cleaned_words (tokens, stop_words): # lemmatize sentence, omit punctuation and stop words such as preposition
    cleaned_tokens = []
    exclusion = ["...", "'"]
    for token, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else: # all the rest tagged with a
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        word = lemmatizer.lemmatize(token, pos)

        if len(word) > 0 and word not in string.punctuation and word.lower() not in stop_words and word not in '...':
            cleaned_tokens.append(word.lower())
    return cleaned_tokens

positive_cleaned_tokens_list=list()
negative_cleaned_tokens_list=list()
for tokens in positive_tokens:
    positive_cleaned_tokens_list.append(cleaned_words(tokens, stop_words))

for tokens in negative_tokens:
    negative_cleaned_tokens_list.append(cleaned_words(tokens, stop_words))

def get_all_words(cleaned_tokens_list):
    all_words = []
    for tokens in cleaned_tokens_list:
        for token in tokens:
            all_words.append(token)
    return all_words

all_neg_words = get_all_words(negative_cleaned_tokens_list)
all_pos_words = get_all_words(positive_cleaned_tokens_list)

def get_dict_for_model(cleaned_tokens_list):
    #value = 'Positive' if valence == 1 else 'Negative'
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_dict_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_dict_for_model(negative_cleaned_tokens_list)

positive_dataset = [(review_dict, "Positive") for review_dict in positive_tokens_for_model]
negative_dataset = [(review_dict, "Negative") for review_dict in negative_tokens_for_model]

import numpy as np
import random

training_no = 5000
positive_train=[]
negative_train=[]
training_set = []

index1 = np.random.choice(len(positive_dataset), training_no)
index2 = np.random.choice(len(negative_dataset), training_no)

positive_train = [positive_dataset[i] for i in index1]
negative_train = [negative_dataset[i] for i in index2]
training_set = positive_train + negative_train