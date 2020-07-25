import pandas as pd # A data manipulation lirrary
import numpy as np # Matrix/Array Computation library
import matplotlib.pyplot as plt # Data Visualization library

# Read the movies and ttarings csv dataset files 
movies_df = pd.read_csv('Datasets/movies.csv')
ratings_df = pd.read_csv('Datasets/ratings.csv')
# Look into the datasets
movies_df.head()

# Cleaning up the datasets by getting year values, removing paranthese, and 
# First 2 lines of code, cleaning up the year values
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
# Removing years from title column and removing blank/null spaces
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()

# Splittinmg the movie title using the sep "|", to organize their titles
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()

moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
        for genre in row['genres']:
                    moviesWithGenres_df.at[index, genre] = 1

# Fill the None type values with the value 0, as the binary values 0 and 1 corresponding to having/not having a category 
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

ratings_df.head()
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()

userinput = [
                    {'title':'Breakfast Club, The', 'rating':5},
                    {'titile':'Toy Story', 'rating':3.5},
                    {'title':'Jumanji', 'rating':2},
                    {'title':"Pulp Fiction", 'rating':5},
                    {'title':'Akira', 'rating':4.5}
            ] 
inputMovies = pd.DataFrame(userinput)
#Filtering out the movies by title
inputd = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movied. t's implicitly merging it by title.
inputMovies = pd.merge(inputd, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()

recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
recommendationTable_df.head()

#The final recommendation table
final_rec = movies_df.loc[movies_df['movied'].isin(recommendationTable_df.head(20).keys())]
print('The final recommendfation is: \n')
print(final_rec)
