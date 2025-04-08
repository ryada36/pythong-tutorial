import numpy as np
from numpy.linalg import norm
import csv
import os

RATINGS_DATASET = os.path.join(os.path.dirname(__file__), 'movies_ratings.csv')
MOVIES_DATASET = os.path.join(os.path.dirname(__file__), 'movies.csv')

user_id_idx = int(input("Enter user id to get recommendations(between [0-5]): "))

parsed_data = []
with open(RATINGS_DATASET, 'r') as f:
    reader = csv.reader(f)
    movies = list(reader)
    movies = movies[1:]  # Skip the header
    for row in movies:
        if row[0] and row[1] and row[2]:  # check not empty
            user_id = int(row[0])
            movie_id = row[1]
            rating = float(row[2])
            parsed_data.append((user_id, movie_id, rating))

user_ids = sorted(list(set([row[0] for row in parsed_data])))
movie_ids = sorted(list(set([row[1] for row in parsed_data])))

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

ratings_matrix = np.zeros((len(user_ids), len(movie_ids)))

# # Fill the matrix
for user_id, movie_name, rating in parsed_data:
    user_idx = user_to_index[user_id]
    movie_idx = movie_to_index[movie_name]
    ratings_matrix[user_idx, movie_idx] = rating

num_users = ratings_matrix.shape[0]

# Initialize similarity matrix
user_similarity = np.zeros((num_users, num_users))

for i in range(num_users):
    for j in range(num_users):
        if i != j:
            u = ratings_matrix[i]
            v = ratings_matrix[j]
            if norm(u) > 0 and norm(v) > 0:  # avoid division by zero
                similarity= np.dot(u, v) / (norm(u) * norm(v))
                user_similarity[i, j] = similarity

print(user_similarity)
def get_top_k_similar_users(user_id, similarity_matrix, k=5):
    user_similarities = similarity_matrix[user_id]
    # Get (user_index, similarity_score) pairs
    similar_users = [(other_user, score) for other_user, score in enumerate(user_similarities) if other_user != user_id]
    # Sort by similarity descending
    similar_users.sort(key=lambda x: x[1], reverse=True)
    # Pick top k
    top_k_users = similar_users[:k]
    return top_k_users


def predict_ratings(user_id, ratings_matrix, top_k_users):
    user_ratings = ratings_matrix[user_id]
    num_movies = ratings_matrix.shape[1]
    
    predicted_ratings = np.zeros(num_movies)
    
    for movie_idx in range(num_movies):
        if user_ratings[movie_idx] == 0:  # If user hasn't rated the movie
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user_id, similarity in top_k_users:
                rating = ratings_matrix[similar_user_id, movie_idx]
                
                if rating != 0:  # Only consider ratings where similar user rated the movie
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_ratings[movie_idx] = weighted_sum / similarity_sum
            else:
                predicted_ratings[movie_idx] = 0  # No similar user rated, keep 0
    
    return predicted_ratings



def recommend_movies(user_id, ratings_matrix, similarity_matrix, movie_ids, k=5, n=5):
    top_k_users = get_top_k_similar_users(user_id, similarity_matrix, k)
    print(f"Top {k} similar users for User {user_id}: {top_k_users}")
    predicted_ratings = predict_ratings(user_id, ratings_matrix, top_k_users)
    # Pick top N movies with highest predicted rating
    recommended_movie_indices = np.argsort(predicted_ratings)[::-1][:n]
    recommended_movies = [movie_ids[idx] for idx in recommended_movie_indices if predicted_ratings[idx] > 0]
    return recommended_movies

recommended = recommend_movies(user_id_idx, ratings_matrix, user_similarity, movie_ids, 3, 3)
# find recommended movies from different movie details csv
recommended_movies = []
with open(MOVIES_DATASET, 'r') as f:
    reader = csv.reader(f)
    movies = list(reader)
    for row in movies:
        if row[0] in recommended:
            recommended_movies.append(row[1])
print(f"Recommended movies for user {user_id_idx}: {recommended_movies}")