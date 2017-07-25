The approach of the problem is as follows:-

1)The main features of the film, namely genre,actor,director,storyline,description,ratings,votes,year are identified.

2)Matrices are formed for features like genre,director,actor,year where [i,j]th entry on each matrix is a measure of similarity between ith and jth movie for the respective feature.

4)An array is formed which stores the rating for each movie.

3)For matrices involving genre,director,actor, the [i,j] th entry is equal to number of genre,director,actor matching between ith and jth movie respectively.[Note: Entry for director matrix can only be 1(matches) or 0(doesnot match) since a film generally has 1 director)

4)For matrix involving year, the number of years between the release of ith and jth movie is calculated.It is then subtracted from the maximum, of the number of years between release of two movies in the dataset(93 in this problem)

5)To calculate ratings of a movie,number of votes is multiplied with the given ratings of the movie to get a general view.

6)Features namely genre,director,actor,year,ratings were then normalized

7)After that the strings 'storyline' and 'descriptions' for each movie is concatenated and stored as 'text'.

8)TfidfVectorizer is used to find similarity between the 'text' for each of the two movies.

9)It returns a matrix 'tfidf_matrix' which a gives a measure of similarity for 'text' between ith and jth movie.

10)Cosine similarity is used to normalize 'tfidf_matrix'

11)Finally weights are given to all the above features and the required sum is computed and stored in the matrix 'final_score'

12)The top 10 final score for each movie is the required output.  

Points to Note:-
1)The weights to the parameters or features are intuitively given. This can be avoided if we are given a training set, from which we can make a linear regression model to find the weights.  
2) The whole algorithm runs in O(n^2) time with O(n^2) extra space since I have used pair wise similarity. We could have improved the efficiency of the code by using clustering on 'storyline'+'description', but it was not giving satisfactory result due to the reason stated below. 

Some of the experiments that were performed on the dataset but failed :-

1)Clustering of movies on 'storyline' + 'description':- I used TfidVectorizer and K-means algorithm to cluster the movies based on 'storyline'+'description' but the results were not satisfactory. The primary reason for this is that the tfidvectorizer creates a similarity score between two texts based on words counts and weightage for each word. It doesnot take the meaning of the text into consideration. So movies with completely different storylines were falling in the same cluster. One possible remedy to this problem is using Word Movers Distance algorithm rather than tfid. This algorithm takes the meaning of the text into account. Due to time constraints I was not able to implement Word Movers Distance algorithm in our problem statement.
 



 
 
