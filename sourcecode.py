
from __future__ import print_function
import numpy as np
import pandas as pd
import math
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import json
import csv
from pprint import pprint

#'data' contains all input information from the file 'input.json'
with open('input.json') as data_file:
	data = json.load(data_file)

#'number_of_movies' stores number of movies in the data list
number_of_movies=len(data)

#'title' stores all movie names
title = []

#'text' is the concatenation of 'description' and 'storyline' for each movie. Later cosine similarity for 'text' between each pair of movie is computed 
text = []

#matrices for features like genre,actor,director,year is made.The (i,j) entry in the matrix is a measure of similarity between ith and jth movie for respective features.
matrix_genre = np.zeros((number_of_movies,number_of_movies))
matrix_actor = np.zeros((number_of_movies,number_of_movies))
matrix_director = np.zeros((number_of_movies,number_of_movies))
matrix_year = np.zeros((number_of_movies,number_of_movies))

#stores a measure of rating for each movie
array_ratings = np.zeros(number_of_movies)

#Storing information in text and title
for field in data:
	title.append(field[u'title'])
	text.append(field[u'description']+field[u'storyline'])


#Storing information in matrix_genre,matrix_actor,matrix_director,array_ratings,matrix_year.For matrix_genre the (i,j)th entry would be equal to number of similar genre between the ith and jth movie.Same for matrix_actor and matrix_director.For array_ratings the ith entry is equal to number of votes for each movie multiplied by movie rating to get more general view.For matrix_year, the number of years between release of two movies has been subtracted from 93 to get a measure of movies of the same period.(maximum gap in years in the dataset between two movies is 93) 

for i in range(number_of_movies):
	for j in range(number_of_movies):
		matrix_genre[i,j] = len((list(set(data[i][u'genre']).intersection(data[j][u'genre']))))
		matrix_actor[i,j] = len((list(set(data[i][u'stars']).intersection(data[j][u'stars']))))
		matrix_director[i,j] = int(((data[i][u'director'])==(data[j][u'director'])))
		matrix_year[i,j]=93-math.fabs(int(data[i][u'year'])-int(data[j][u'year']))
		array_ratings[i]=float(data[i][u'votes'].replace(',', ''))*float(data[i][u'rating'])

# all parameters have been normalized to remove biasness.

matrix_genre = (matrix_genre-(np.sum(matrix_genre))/(number_of_movies*number_of_movies))/(np.amax(matrix_genre)-np.amin(matrix_genre))
matrix_actor = (matrix_actor-(np.sum(matrix_actor))/(number_of_movies*number_of_movies))/(np.amax(matrix_actor)-np.amin(matrix_actor))
matrix_director = (matrix_director-(np.sum(matrix_director))/(number_of_movies*number_of_movies))/(np.amax(matrix_director)-np.amin(matrix_director))
matrix_year = (matrix_year-(np.sum(matrix_year))/(number_of_movies*number_of_movies))/(np.amax(matrix_year)-np.amin(matrix_director))
array_ratings =(array_ratings-(np.sum(array_ratings))/(number_of_movies))/(np.amax(array_ratings)-np.amin(array_ratings))


#Below is the code for text clustering 
# load nltk's English stopwords as variable called 'stopwords'

stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

#function to tokenize and stem words in the parameter 'text' and return the stemmed words.

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(text)

#store cosine similarity between each pair of movie based on text clustering on movie description and storyline
from sklearn.metrics.pairwise import cosine_similarity
dist = cosine_similarity(tfidf_matrix)


#the (i,j)th entry in final score is a measure of similarity between ith and jth movie.
final_score = np.zeros((number_of_movies,number_of_movies))

#It is the final_list in which top 10 similar movie indexes for each movie is stored
final_list = []

#Weightage for different features to get final score
text_weightage = .3
genre_weightage = .2
actor_weightage = .2
director_weightage = .15
ratings_weightage= .1
ratings_year = .05

for i in range(number_of_movies):
	for j in range(number_of_movies):
		final_score[i,j]=(text_weightage*dist[i][j])+(genre_weightage*matrix_genre[i,j])+(actor_weightage*matrix_actor[i,j])+(director_weightage*matrix_director[i,j])+(ratings_weightage*array_ratings[i])+(ratings_year*matrix_year[i,j])

#since similar movies cannot contain the movie itself
for i in range(number_of_movies):
	final_score[i,i] =-2

#get top 10 similar movie indexes for each movie from final_score
for i in range(number_of_movies):
	final_list.append(final_score[i].argsort()[-10:][::-1])

#get movie names from indexes in 'final_list' and store it in 'output' dictionary
output = {}

for i in range(number_of_movies):
	output[data[i][u'title']] = []
	for j in range(len(final_list[i])):
		output[data[i][u'title']].append(data[final_list[i][j]][u'title'])

#write 'output' to a json file
with open('output.json', 'w') as fp:
    json.dump(output, fp)








