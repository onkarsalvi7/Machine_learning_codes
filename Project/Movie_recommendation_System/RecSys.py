
"""
@author: Onkar
"""

#MOVIE RECOMMENDATION SYSTEM
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tkinter


Movies = pd.read_csv('movies.csv')
Ratings = pd.read_csv('ratings.csv')
Tags = pd.read_csv('tags.csv')
Links = pd.read_csv('links.csv')


#Getting all the movies watched by each user
def get_movies_by_user(ratings):
    '''
    get all the movies watched by each user
    -----
    Input: pandas dataframe with features [userId, movieId, rating, timestamp]
    -----
    Output: A dictionary with users as the key and the movies that each user watched as values
    ------
    '''
    user_movies = {}
    for user in ratings['userId']:
        user_movies[user] = []
    movies = ratings['movieId'].values
    for i, user  in enumerate(ratings['userId']):
        user_movies[user].append(movies[i])
    
    return user_movies
        

#Getting the genres of each movie in a pandas Dataframe
def get_genres(movies):
    '''
    gets the genres of the movies in the dataset
    -----
    Input: pandas dataframe with features [movieId, title, genres]
    -----
    Output: A pandas dataframe with features [movieId, genre_1,..., genre_N] and 
    a value 1 for the genre which the film belongs to and 0 elsewhere
    ------
    '''  
    def split_genres(genres):
        genre = [word for word in genres.split('|')]
        return genre
    
    movies['Genres_split'] = movies['genres'].apply(lambda x: split_genres(x))
    G = []
    for gen in movies['Genres_split']:
        for g in gen:
            if g not in G:
                G.append(g)
    gen_frame = pd.DataFrame(np.zeros((len(movies), len(G))), columns = G)
    for i, m in enumerate(movies['Genres_split']):
        for gen in m:
            gen_frame.at[i, gen] = 1
    
    frame = pd.concat([movies['movieId'], gen_frame], axis = 1)
    return frame



def get_rating(ratings, movies):
    '''
    get the average user rating for each movie
    -----
    Input: 
    ratings: pandas dataframe with features [userId, movieId, rating, timestamp]
    movies: pandas dataframe with features [movieId, title, genres]
    -----
    Output: A dataframe with features [movieId , ave_rating]
    ------
    '''
    mov_ratings = {}
    movie = ratings['movieId'].values
    rating = ratings['rating'].values
    
    #getting all the ratings for each movie 
    for mov_id in movie:
        if mov_id not in mov_ratings:
            mov_ratings[mov_id] = []
    
    for i, mov_id in enumerate(movie):
        mov_ratings[mov_id].append(rating[i])
        
    rat = np.zeros(len(movies))
    
    #Handling the missing movies
    for mov in movies['movieId'].values:
        if mov not in mov_ratings:
            mov_ratings[mov] = [0]
            
    #Getting the average ratings for each movie
    for i, mov in enumerate(movies['movieId'].values):
        #print(mov_ratings[i])
        rat[i] = sum(mov_ratings[mov])/len(mov_ratings[mov])
        
    movie_ratings = pd.DataFrame(rat, index = movies['movieId'], columns = ['avg_rating'])
    return movie_ratings

        
def get_movies_dataframe(groups):
    data = np.zeros((len(groups), len(groups)))
    df = pd.DataFrame(data, index = groups, columns = groups)
    return df


def get_support(movieid, user_movies):
    counter = 0
    for user in user_movies:
        for mov in user_movies[user]: 
            if mov==movieid:
                counter += 1
    support = counter/len(user_movies)
    return support



def get_support_confidence(movieid1,movieid2, user_movies):
    counter1 = 0
    
    user1 = []
    for user in user_movies:
        for mov in user_movies[user]: 
            if mov == movieid1:
                counter1 += 1
                user1.append(user)
                
    support = counter1/len(user_movies)
    
    counter2 = 0
    for user in user1:
        for mov in user_movies[user]:
            if mov == movieid2:
                counter2 += 1
    
    confidence = counter2/counter1
    return support, confidence

def get_lift(movieid1, movieid2, user_movies):
    support2, confidence1_2 = get_support_confidence(movieid1, movieid2, user_movies)
    if support2 == 0:
        return 0
    lift = confidence1_2/support2
    return lift


def train_apriori(df, user_movies, movieid):
    lift_table = []
    count = 0
    for mov2 in df.columns.values:
        lift_table.append(get_lift(movieid,mov2, user_movies))
        count += 1
        
    lift_table = np.asarray(lift_table)
    table = pd.DataFrame(lift_table, index = df.index, columns = ['lift_score'])
    return table


def get_clusters(movies, y):
    cluster = []
    for i in range(100):
        c = movies['movieId'].values[y==i].tolist()
        cluster.append(c)
    return cluster

def get_group(movieid, cluster):
    for clus in range(len(cluster)):
        if movieid in cluster[clus]:
            break
        
    return cluster[clus]

def id_to_movie(movies, rec_movid):
    mov_names = []
    for mov in movies['movieId']:
        if mov in rec_movid:
            mov1 = movies['title'].values[movies['movieId']==mov]
            mov_names.append(mov1.tolist())
    
    return mov_names
            
def get_movie_id(movies, movie_title):
    return movies['movieId'][movies['title']==movie_title].values[0]
        



OPTIONS = Movies['title'].values.tolist()

def get_input():
    get_mov = tkinter.Tk()
    get_mov.title('Movie Recommender')
    
    frame = tkinter.Frame(get_mov, width=55, height = 35)
    frame.pack()
    
    #getting the drop down list of movies
    variable = tkinter.StringVar(get_mov)
    variable.set('initial value')
    while(variable.get()!="Choose a movie"):
        variable.set("Choose a movie")
        w = tkinter.OptionMenu(get_mov, variable, *OPTIONS)
        w.pack()
    
    #for the OK button
    def ok():
        get_mov.destroy()
    button = tkinter.Button(get_mov, text="OK", command = ok)
    button.pack()
    
    get_mov.mainloop()
    input_movie = variable.get()
    input_id = get_movie_id(Movies, input_movie)
    return input_id

def show_output(names):
    show = tkinter.Tk()
    show.title("Movies Recommended for you ")
    frame = tkinter.Frame(show, width=55, height = 35)
    frame.pack()
    
    w = tkinter.Listbox(show, width = 50, height = 30)
    w.insert(1, names[4][0])
    w.insert(2, names[3][0])
    w.insert(3, names[2][0])
    w.insert(4, names[1][0])
    w.insert(5, names[0][0])
    w.pack()
    
    show.mainloop()

def main():
    #getting all the movies watched by each user
    user_movies = get_movies_by_user(Ratings)
    
    #Getting average rating for each movie
    avg_rating = get_rating(Ratings, Movies)
    
    #generate an encoder for the genres of the movie
    gen_frame = get_genres(Movies)
    
    #Getting genres and ratings in the features for Kmeans clustering
    features = np.append(avg_rating.values, gen_frame.values[:, 1:], axis = 1)
    
    #Creating an object to form 100 clusters using K-Means Clustering
    kmc = KMeans(n_clusters = 100).fit(features)     
    
    #forming clusters of data        
    y = kmc.predict(features)
    
    #Getting all the clusters together
    cluster = get_clusters(Movies, y)
    
    #Getting the input movie
    N = get_input()
    
    #Getting the cluster for the input movie
    group = get_group(N, cluster)
    
    #Create a dataframe using the movies in the current cluster for apriori
    df = get_movies_dataframe(group)
    
    #Train the apriori for the selected movie
    table = train_apriori(df, user_movies, N) 
    
    #Sort in increasing order of lift value
    sorted_table = table.sort_values('lift_score')
    
    #Pick only top 5 movies with the hoghest lift value
    rec_movies = np.asarray(sorted_table.index[-6:-1].values.tolist())  
    
    names = id_to_movie(Movies, rec_movies)
    
    show_output(names)
        
        
if __name__== "__main__":
    main()
    
    
    
    
    
    
