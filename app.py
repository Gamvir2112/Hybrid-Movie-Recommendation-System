from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from difflib import get_close_matches
import os
import urllib.request
import zipfile
import pickle
from flask_cors import CORS

# Initialize Flask app with static folder for serving HTML
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Initialize global recommender object
recommender = None

# Configuration - your specific model path
MODEL_PATH = r"C:\Users\ACER\Final Year Project\project\model.pkl"
DATA_DIR = os.environ.get('DATA_DIR', 'data')

def initialize_recommender():
    """Initialize and load the recommender system from the specific model path"""
    global recommender
    
    # Try to load from the specific pickle file path
    try:
        print(f"Attempting to load model from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as file:
            recommender = pickle.load(file)
        print("Recommender system loaded from file successfully!")
        return
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Could not load model from file: {str(e)}")
        print("Building new recommender model...")
    
    
    
    # Save the newly created model to the specified path
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(recommender, file)
        print(f"New recommender model saved to {MODEL_PATH}!")
    except Exception as e:
        print(f"Warning: Could not save model to file: {str(e)}")
    
    print("Recommender system initialized successfully!")

# Serve the HTML file at the root URL
@app.route('/')
def index():
    return app.send_static_file('index.html')

# API routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommender not initialized"}), 500
    return jsonify({"status": "ok", "message": "Recommender system is running"}), 200

@app.route('/api/search_movie', methods=['GET'])
def search_movie():
    """Search for a movie by title (with fuzzy matching)"""
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommender not initialized"}), 500
    
    query = request.args.get('title', '')
    if not query:
        return jsonify({"status": "error", "message": "No movie title provided"}), 400
    
    try:
        matched_title = recommender.find_movie_by_title(query)
        matched_idx = recommender.movie_idx[matched_title]
        matched_genres = recommender.movies.iloc[matched_idx]['genres']
        
        # Log the match for debugging
        print(f"User searched for: '{query}', matched to: '{matched_title}'")
        
        # Find the corresponding movie ID
        movie_id = next(key for key, value in recommender.movie_titles.items() if value == matched_title)
        
        return jsonify({
            "status": "success", 
            "movie": {
                "id": int(movie_id),
                "title": matched_title,
                "genres": matched_genres
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get movie recommendations based on a movie title"""
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommender not initialized"}), 500
    
    query = request.args.get('title', '')
    method = request.args.get('method', 'hybrid')
    count = request.args.get('count', '5')
    
    if not query:
        return jsonify({"status": "error", "message": "No movie title provided"}), 400
    
    try:
        count = int(count)
        count = max(1, min(20, count))  # Limit between 1 and 20
    except:
        count = 5  # Default if invalid
    
    # Validate method
    valid_methods = ['hybrid', 'collaborative', 'content', 'matrix']
    if method not in valid_methods:
        method = 'hybrid'  # Default to hybrid if invalid
    
    try:
        # First, find the matched movie
        matched_title = recommender.find_movie_by_title(query)
        matched_idx = recommender.movie_idx[matched_title]
        matched_genres = recommender.movies.iloc[matched_idx]['genres']
        
        # Find the corresponding movie ID
        movie_id = next(key for key, value in recommender.movie_titles.items() if value == matched_title)
        
        # Get recommendations based on method
        if method == 'hybrid':
            recommendation_ids = recommender.get_hybrid_recommendations(matched_title, n_recommendations=count)
            recommendation_titles = recommendation_ids  # already contains titles
        elif method == 'collaborative':
            movie_ids = recommender.find_similar_movies_collaborative(movie_id, k=count)
            recommendation_titles = [recommender.movie_titles[mid] for mid in movie_ids]
        elif method == 'content':
            recommendation_titles = recommender.get_content_based_recommendations(matched_title, n_recommendations=count)
            recommendation_titles = list(recommendation_titles)  # Convert from pandas Series to list
        elif method == 'matrix':
            movie_ids = recommender.get_matrix_factorization_recommendations(movie_id, k=count)
            recommendation_titles = [recommender.movie_titles[mid] for mid in movie_ids]
        
        # Get details for each recommended movie
        recommendations = []
        for title in recommendation_titles:
            # Find the movie ID
            rec_id = next(key for key, value in recommender.movie_titles.items() if value == title)
            # Find the index in the DataFrame
            rec_idx = recommender.movies[recommender.movies['movieId'] == rec_id].index[0]
            # Get the genres
            rec_genres = recommender.movies.iloc[rec_idx]['genres']
            
            recommendations.append({
                "id": int(rec_id),
                "title": title,
                "genres": rec_genres
            })
        
        return jsonify({
            "status": "success",
            "input_movie": {
                "id": int(movie_id),
                "title": matched_title,
                "genres": matched_genres
            },
            "method": method,
            "recommendations": recommendations
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/top_movies', methods=['GET'])
def get_top_movies():
    """Get top rated movies using Bayesian average"""
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommender not initialized"}), 500
    
    try:
        min_ratings = request.args.get('min_ratings', '50')
        count = request.args.get('count', '10')
        
        try:
            min_ratings = int(min_ratings)
            min_ratings = max(1, min_ratings)  # Ensure positive value
        except:
            min_ratings = 50  # Default if invalid
        
        try:
            count = int(count)
            count = max(1, min(50, count))  # Limit between 1 and 50
        except:
            count = 10  # Default if invalid
        
        # Get top rated movies
        top_movies_df = recommender.get_top_rated_movies(min_ratings=min_ratings)
        
        # Convert to list of dictionaries with required fields
        top_movies = []
        for _, row in top_movies_df.head(count).iterrows():
            # Find the index in the DataFrame
            movie_idx = recommender.movies[recommender.movies['movieId'] == row['movieId']].index[0]
            # Get the genres
            genres = recommender.movies.iloc[movie_idx]['genres']
            
            top_movies.append({
                "id": int(row['movieId']),
                "title": row['title'],
                "rating": float(row['bayesian_avg']),
                "num_ratings": int(row['num_ratings']),
                "genres": genres
            })
        
        return jsonify({
            "status": "success",
            "min_ratings": min_ratings,
            "top_movies": top_movies
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get list of all movie genres"""
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommender not initialized"}), 500
    
    try:
        # Extract all unique genres
        all_genres = set()
        for genres in recommender.movies['genres']:
            all_genres.update(genres)
        
        # Count movies per genre
        genre_counts = {}
        for genre in all_genres:
            count = sum(1 for genres in recommender.movies['genres'] if genre in genres)
            genre_counts[genre] = count
        
        # Sort by count (descending)
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            "status": "success",
            "genres": [{"name": genre, "count": count} for genre, count in sorted_genres]
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/save_model', methods=['GET'])
def save_model():
    """Save the current recommender model to a pickle file"""
    if recommender is None:
        return jsonify({"status": "error", "message": "No recommender to save"}), 500
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(recommender, file)
        return jsonify({
            "status": "success", 
            "message": f"Model saved successfully to {MODEL_PATH}"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to save model: {str(e)}"
        }), 500

@app.route('/api/download_model', methods=['GET'])
def download_model():
    """Download the current model file"""
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "status": "error", 
            "message": "Model file does not exist"
        }), 404
    
    try:
        return send_file(
            MODEL_PATH,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='movie_recommender_model.pkl'
        )
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to download model: {str(e)}"
        }), 500

class MovieRecommender:
    def __init__(self, ratings_path, movies_path):
        """
        Initialize the MovieRecommender with paths to the data files.
        
        Parameters:
        ratings_path (str): Path to the ratings CSV file
        movies_path (str): Path to the movies CSV file
        """
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.X = None
        self.user_mapper = None
        self.movie_mapper = None
        self.user_inv_mapper = None
        self.movie_inv_mapper = None
        self.movie_titles = None
        self.movie_idx = None
        self.cosine_sim = None
        
        # Process data immediately
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Perform initial data preprocessing steps:
        1. Create utility matrix
        2. Process movie genres
        3. Create mappers for easy lookup
        """
        # Extract movie title dictionary for easier lookup
        self.movie_titles = dict(zip(self.movies['movieId'], self.movies['title']))
        
        # Process genres to convert from string to list
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split('|'))
        
        # Create utility matrix for collaborative filtering
        self.create_utility_matrix()
        
        # Create genre features for content-based filtering
        self.create_genre_features()
        
        # Create movie index dictionary for content-based recommendations
        self.movie_idx = dict(zip(self.movies['title'], list(self.movies.index)))
    
    def create_utility_matrix(self):
        """Create sparse utility matrix for collaborative filtering"""
        # Get unique counts
        n_users = self.ratings['userId'].nunique()
        n_movies = self.ratings['movieId'].nunique()
        
        # Create user and movie mappers
        self.user_mapper = dict(zip(np.unique(self.ratings["userId"]), list(range(n_users))))
        self.movie_mapper = dict(zip(np.unique(self.ratings["movieId"]), list(range(n_movies))))
        
        # Create inverse mappers
        self.user_inv_mapper = dict(zip(list(range(n_users)), np.unique(self.ratings["userId"])))
        self.movie_inv_mapper = dict(zip(list(range(n_movies)), np.unique(self.ratings["movieId"])))
        
        # Map indices
        user_index = [self.user_mapper[i] for i in self.ratings['userId']]
        movie_index = [self.movie_mapper[i] for i in self.ratings['movieId']]
        
        # Create sparse matrix
        self.X = csr_matrix((self.ratings["rating"], (user_index, movie_index)), shape=(n_users, n_movies))
    
    def create_genre_features(self):
        """Create binary features for each genre"""
        # Extract all unique genres
        all_genres = set()
        for genres in self.movies['genres']:
            all_genres.update(genres)
        
        # Create binary columns for each genre
        for genre in all_genres:
            self.movies[genre] = self.movies['genres'].apply(lambda x: 1 if genre in x else 0)
        
        # Create genre features matrix
        self.movie_genres = self.movies.drop(columns=['movieId', 'title', 'genres'])
        
        # Calculate cosine similarity between movies based on genres
        self.cosine_sim = cosine_similarity(self.movie_genres, self.movie_genres)

    def get_matrix_sparsity(self):
        """Calculate and return the sparsity of the utility matrix"""
        n_users, n_movies = self.X.shape
        n_total = n_users * n_movies
        n_ratings = self.X.nnz
        sparsity = n_ratings / n_total
        return sparsity
    
    def find_similar_movies_collaborative(self, movie_id, k=10, metric='cosine'):
        """
        Find similar movies using collaborative filtering
        
        Parameters:
        movie_id (int): ID of the movie to find similar movies for
        k (int): Number of similar movies to return
        metric (str): Distance metric for kNN calculations
        
        Returns:
        list: List of similar movie IDs
        """
        # Transpose X to get movie-user matrix
        X = self.X.T
        neighbor_ids = []
        
        # Get movie index and vector
        movie_idx = self.movie_mapper[movie_id]
        movie_vec = X[movie_idx]
        
        # Reshape if necessary
        if isinstance(movie_vec, (np.ndarray)):
            movie_vec = movie_vec.reshape(1, -1)
        
        # Create and fit kNN model
        knn = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
        knn.fit(X)
        
        # Get nearest neighbors
        neighbors = knn.kneighbors(movie_vec, return_distance=False)
        
        # Extract neighbor IDs
        for i in range(0, k+1):
            n = neighbors.item(i)
            neighbor_ids.append(self.movie_inv_mapper[n])
        
        # Remove the original movie from results
        if movie_id in neighbor_ids:
            neighbor_ids.remove(movie_id)
        else:
            neighbor_ids = neighbor_ids[:k]
            
        return neighbor_ids
    
    def find_movie_by_title(self, title_query):
        """
        Find the closest movie title match using improved matching algorithm
        
        Parameters:
        title_query (str): Approximate movie title to search for
        
        Returns:
        str: Best matching movie title
        """
        all_titles = self.movies['title'].tolist()
        
        # Normalize the title query (convert to lowercase)
        title_query_norm = title_query.lower()
        
        # Special case for "godfather" or "god father"
        if "godfather" in title_query_norm or "god father" in title_query_norm:
            godfather_titles = [t for t in all_titles if "godfather" in t.lower()]
            if godfather_titles:
                return godfather_titles[0]
        
        # Try direct case-insensitive substring matching first
        direct_matches = [title for title in all_titles if title_query_norm in title.lower()]
        if direct_matches:
            # Sort by length to prefer shorter matches (more specific)
            return sorted(direct_matches, key=len)[0]
        
        # Try word-by-word matching (for cases like "star wars" vs "Star Wars: Episode IV")
        query_words = title_query_norm.split()
        if len(query_words) > 1:
            word_matches = []
            for title in all_titles:
                title_lower = title.lower()
                if all(word in title_lower for word in query_words):
                    word_matches.append(title)
            
            if word_matches:
                return sorted(word_matches, key=len)[0]
        
        # Then try fuzzy matching with a lower threshold for better matches
        closest_matches = get_close_matches(title_query, all_titles, n=1, cutoff=0.4)
        if closest_matches:
            return closest_matches[0]
        
        # If still no match, return first title as fallback
        return all_titles[0]
    
    def get_content_based_recommendations(self, title_query, n_recommendations=10):
        """
        Get movie recommendations based on content similarity (genres)
        
        Parameters:
        title_query (str): Approximate movie title to get recommendations for
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        pd.Series: Series of recommended movie titles
        """
        # Find the closest match to the query
        title = self.find_movie_by_title(title_query)
        
        # Get the movie index
        idx = self.movie_idx[title]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendation indices
        sim_scores = sim_scores[1:(n_recommendations+1)]
        movie_indices = [i[0] for i in sim_scores]
        
        # Return movie titles
        return self.movies['title'].iloc[movie_indices]
    
    def get_top_rated_movies(self, min_ratings=50):
        """
        Get top rated movies using Bayesian average
        
        Parameters:
        min_ratings (int): Minimum number of ratings for a movie to be considered
        
        Returns:
        pd.DataFrame: DataFrame with top rated movies
        """
        # Group ratings by movie
        movie_stats = self.ratings.groupby('movieId').agg(
            mean_rating=('rating', 'mean'),
            num_ratings=('rating', 'count')
        ).reset_index()
        
        # Filter by minimum number of ratings
        movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]
        
        # Calculate Bayesian average parameters
        C = movie_stats['num_ratings'].mean()
        m = movie_stats['mean_rating'].mean()
        
        # Apply Bayesian average formula
        movie_stats['bayesian_avg'] = (C * m + movie_stats['num_ratings'] * movie_stats['mean_rating']) / (C + movie_stats['num_ratings'])
        
        # Merge with movie titles
        movie_stats = movie_stats.merge(self.movies[['movieId', 'title']])
        
        # Sort by Bayesian average
        return movie_stats.sort_values('bayesian_avg', ascending=False)
    
    def get_matrix_factorization_recommendations(self, movie_id, n_components=20, n_iter=10, k=10):
        """
        Get movie recommendations using matrix factorization
        
        Parameters:
        movie_id (int): ID of the movie to find similar movies for
        n_components (int): Number of latent factors
        n_iter (int): Number of iterations for SVD
        k (int): Number of recommendations to return
        
        Returns:
        list: List of similar movie IDs
        """
        # Apply Truncated SVD
        svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=42)
        Q = svd.fit_transform(self.X.T)
        
        # Create mapper dictionaries for the factor matrix
        movie_mapper_svd = {self.movie_inv_mapper[i]: i for i in range(len(self.movie_inv_mapper))}
        movie_inv_mapper_svd = {i: self.movie_inv_mapper[i] for i in range(len(self.movie_inv_mapper))}
        
        # Find similar movies in the factor space
        knn = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric="cosine")
        knn.fit(Q)
        
        movie_idx = movie_mapper_svd[movie_id]
        movie_vec = Q[movie_idx].reshape(1, -1)
        
        neighbors = knn.kneighbors(movie_vec, return_distance=False)
        
        neighbor_ids = []
        for i in range(1, k+1):  # Skip the first one (the movie itself)
            n = neighbors.item(i)
            neighbor_ids.append(movie_inv_mapper_svd[n])
            
        return neighbor_ids
    
    def get_hybrid_recommendations(self, movie_id_or_title, n_recommendations=10):
        """
        Get hybrid recommendations combining collaborative and content-based filtering
        
        Parameters:
        movie_id_or_title: Either a movie ID (int) or movie title (str)
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        list: List of recommended movie titles
        """
        # Determine if input is a movie ID or title
        if isinstance(movie_id_or_title, int) or (isinstance(movie_id_or_title, str) and movie_id_or_title.isdigit()):
            movie_id = int(movie_id_or_title)
            movie_title = self.movie_titles[movie_id]
        else:
            movie_title = self.find_movie_by_title(movie_id_or_title)
            movie_id = next(key for key, value in self.movie_titles.items() if value == movie_title)
        
        # Get collaborative filtering recommendations
        collab_movies = self.find_similar_movies_collaborative(movie_id, k=n_recommendations)
        
        # Get content-based recommendations
        content_titles = self.get_content_based_recommendations(movie_title, n_recommendations)
        content_movies = [key for key, value in self.movie_titles.items() if value in content_titles.values]
        
        # Combine recommendations with weights
        # Movies that appear in both lists get higher weight
        all_movies = list(collab_movies) + list(content_movies)
        movie_counts = Counter(all_movies)
        
        # Sort by count (prioritizing movies that appear in both lists)
        # Then by a random factor to break ties consistently
        np.random.seed(42)
        movie_scores = {movie: (count, np.random.random()) for movie, count in movie_counts.items()}
        
        # Sort and get top N movies
        recommended_movies = sorted(movie_scores.keys(), 
                                   key=lambda x: (movie_scores[x][0], movie_scores[x][1]), 
                                   reverse=True)[:n_recommendations]
        
        # Return movie titles
        return [self.movie_titles[movie_id] for movie_id in recommended_movies]

# Initialize the recommender system when the server starts
if __name__ == '__main__':
    initialize_recommender()
    app.run(debug=True, host='0.0.0.0', port=5000)
