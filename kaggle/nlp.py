import numpy as np
import pandas as pd
import spacy


nlp = spacy.load('en')


### Tokenizing & Text Preprocessing
doc = nlp("Tea is healthy and calming, don't you think?")
print('Token \t\t\tLemma \t\t\tStopword')
print('-'*60)
for token in doc:
    '''
    .lemma_ returns the base form of the word
    Lemmatizing can be used to combine multiple forms of the same word into one base form
    
    .is_stop returns whether the word occurs frequently in the English language
    and does not contain valuable information e.g. "the", "is", "and", "but"

    Note: Lemmatizing and dropping stopwords might worsen your model, and thus should be treated 
            this preprocessing as part of the hyperparameter optimization process. 
    '''
    print(f"{token} \t\t\t{token.lemma_} \t\t\t{token.is_stop}")

### Pattern Matching
from spacy.matcher import PhraseMatcher


# Matcher efficiently match terminology lists.
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
# lemma = [token.lemma_ for pattern in patterns for token in pattern]

'''
    Adds a match-rule to the phrase-matcher
    
    .add('id key', 'on-match callback', *patterns)
    patterns is a list of `doc` objects representing match patterns
    *patterns is packed into a tuple which is immutable in contrast to list which is mutable
'''
matcher.add("terminology_list", None, *patterns)

text_doc = nlp("Glowing review overall, and some really interesting side-by-side photography tests pitting the 11 Pro against the Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 

# matcher finds terms that occurs in the text
# returns a tuple of (match_id, start, end)
# match_id is the same for every item that comes in the same terminology list
matches = matcher(text_doc)

# Example:
match_id, start, end = matches[0]
# print(nlp.vocab.strings[match_id], text_doc[start:end])

### Text classification

# Bag of Words represents the count of each term's occurance in a document 
# TODO: Checkout TF-IDF (Term Frequency - Inverse Document Frequency). Term count is scaled by term's frequency in the corpus.

# Create text categorizer from scratch
text_categorizer = nlp.create_pipe(
                        'textcat', 
                        config={
                            'exclusive_classes': True, # True because it can be either "ham" or "spam"
                            'architecture': 'bow'}) # bow = bag of words architecture (spacy has neural network available)

# Add text categorizer to an empty nlp model
nlp.add_pipe(text_categorizer)
# Add label to text classifier
text_categorizer.add_label('ham')
text_categorizer.add_label('spam')

# Prepare training and label data 
spam = pd.read_csv('/kaggle/input/nlp-course/spam.csv')
train_texts = spam['text'].values
train_labels = [{'cats': {'ham': label == 'ham', 'spam': label == 'spam'}} for label in spam['label']]
train_data = list(zip(train_texts, train_labels))
train_data[:3]

# Training the model
from spacy.util import minibatch
import random

# Fix seed for reproducibility
spacy.util.fix_random_seed(1)
random.seed(1)
optimizer = nlp.begin_training()
losses = {}
for epoch in range(1):
    random.shuffle(train_data)
    # Create minibatches of data to make training more efficient
    batches = minibatch(train_data, size=8)
    for batch in batches:
        texts, labels = zip(*batch)
        '''
        Updates the model in the pipeline
        https://spacy.io/api/language#update
        '''
        nlp.update(texts, labels, sgd=optimizer, losses=losses)

    print(losses)

# Testing the model 
texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA" ]

docs = [nlp.tokenizer(text) for text in texts] # Input text must be first tokenized with nlp.tokenizer
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])

### Word Vectors

# Word embeddings can represent each world numerically in such a way that the vector corresponds to how that word is used or what it means.
# It considers the context in which the words appear 

# spaCy provide embeddings from model called Word2Vec
# This is available from large language model called 'en_core_web_lg'
nlp = spacy.load('en_core_web_lg')
text = "These vectors can be used as features for machine learning models."
with nlp.disable_pipes(): # disabling pipes for efficiency
    vectors = np.array([token.vector for token in nlp(text)])

### Document similarity 

# Documents with similar contents will generally have similar vectors. 
# A common metric to determine the similarity of documents is called the cosine similarity

def cosine_similarity(a, b):
    return a.dot(b) / np.sqrt(a.dot(a) * b.dot(b))

a = nlp("REPLY NOW FOR FREE TEA").vector
b = nlp("According to legend, Emperor Shen Nung discovered tea when leaves from a wild tree blew into his pot of boiling water.").vector
cosine_similarity(a, b)


# Intro to NLP Practice
# Problem: Determine which dishes people liked and disliked. 

data = pd.read_json('/kaggle/input/nlp-course/restaurant.json')

menu_terms = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
            "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
            "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
            "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
            "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
            "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
            "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
            "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  
             "Prosciutto", "Salami"]

menu_terms_tokenized = [nlp(term) for term in menu_terms]
matcher.add('menu_term_list', None, *menu_terms_tokenized)

# Ratings are procured for each item in the menu

from collections import defaultdict

# item_ratings is a dictionary of lists
item_ratings = defaultdict(list)

# iterrows iterate over dataframe as (index, Series) pairs
for idx, review in data.iterrows():
    doc = nlp(review.text)
    matches = matcher(doc)
    # find unique terms that match to the doc 
    found_items = set([doc[match[1]:match[2]] for match in matches])
    
    for item in found_items:
        item_ratings[str(item).lower()].append(review.stars)

# Quick stats on item ratings
# .items returns key-pair value, .keys returns only keys, and .values returns only the values
mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}
# key=mean_ratings.get means to also return the key associated to the value 
sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)

# Quick stats on item counts
counts = {item: len(ratings) for item, ratings in item_ratings.items()}
item_counts = sorted(counts, key=counts.get, reverse=True)
for item in item_counts:
    print(f"{item:>25}{counts[item]:>5}")
    
# Result
print("Worst rated menu items:")
for item in sorted_ratings[:10]:
    print(f"{item:20} Average rating:{mean_ratings[item]:.2f} \tcount: {counts[item]}")

print("Best rated menu items:")
for item in sorted_ratings[-10:]:
    print(f"{item:20} Average rating:{mean_ratings[item]:.2f} \tcount: {counts[item]}")
    
# The less data you have for any specific item, the less you can trust that the average rating is the "real" sentiment of the customers. 
# As the number of data points increases, the error on the mean decreases as 1 / sqrt(n).

### Notes

# For splitting training and testing datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(*arrays, **options)
'''
Command options:
    > test_size: 0.0 - 1.0, represents the prop of dataset to include in train split 
    > random_state: pass an int for reproducible output
    > shuffle: boolean, determines whether data should be shuffled before splitting
'''

# SVM classifier
from sklearn.svm import LinearSVC

# Set dual=False to speed up training, and it's not needed
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )