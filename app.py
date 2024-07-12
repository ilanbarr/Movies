from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
import requests
from bs4 import BeautifulSoup

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Define User and Preference models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Preference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables explicitly
with app.app_context():
    db.create_all()

# Load datasets and model
df_ratings = pd.read_csv('ratings.csv')
df_genome_scores = pd.read_csv('genome-scores.csv')
df_movies = pd.read_csv('movies.csv')
df_features = df_genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
genome_movie_ids = set(df_genome_scores['movieId'].unique())

df_link = pd.read_csv('links.csv')  
df_movies = pd.read_csv('movies.csv')

df_merged_movies = pd.merge(df_link, df_movies, on='movieId')

with open('svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)

def get_poster_url(tmdb_id):
    url = f'https://www.themoviedb.org/movie/{int(tmdb_id)}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        poster_img = soup.find('img', class_='poster w-full')
        if poster_img and 'src' in poster_img.attrs:
            return poster_img['src']
    return None

def hybrid_recommendation(user_id, top_n=10):
    user_preferences = Preference.query.filter_by(user_id=user_id).all()
    liked_movie_ids = [pref.movie_id for pref in user_preferences]
    
    if not liked_movie_ids:
        return [("No preferences found for this user", None, None)]

    user_rated_movies = df_ratings[df_ratings['userId'] == user_id]['movieId'].tolist()
    predictions = [svd.predict(user_id, mid) for mid in df_movies['movieId'] if mid not in user_rated_movies]
    cf_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
    cf_movie_ids = [pred.iid for pred in cf_recommendations]

    valid_liked_movie_ids = [mid for mid in liked_movie_ids if mid in genome_movie_ids]
    
    if valid_liked_movie_ids:
        liked_movie_features = df_features.loc[valid_liked_movie_ids].mean(axis=0)
        content_predictions = cosine_similarity([liked_movie_features], df_features)
        content_movie_indices = content_predictions.argsort()[0][::-1]
        content_movie_ids = [df_features.index[idx] for idx in content_movie_indices if df_features.index[idx] not in liked_movie_ids][:top_n]
    else:
        content_movie_ids = []

    combined_movie_ids = list(set(cf_movie_ids + content_movie_ids))
    combined_movie_ids = [mid for mid in combined_movie_ids if mid not in liked_movie_ids][:top_n]
    
    recommended_movies = []
    for mid in combined_movie_ids:
        title = df_movies[df_movies['movieId'] == mid]['title'].values[0]
        tmdb_id = df_merged_movies[df_merged_movies['movieId'] == mid]['tmdbId'].values[0]
        imdb_id = df_merged_movies[df_merged_movies['movieId'] == mid]['imdbId'].values[0]
        imdb_id_formatted = f"{int(imdb_id):07d}"
        poster_url = get_poster_url(tmdb_id)
        recommended_movies.append((title, poster_url, imdb_id_formatted))
    
    return recommended_movies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    try:
        liked_movie_ids = request.form.getlist('movies')
        logging.debug(f"Received liked movie IDs: {liked_movie_ids}")
        
        liked_movie_ids = [int(movie_id) for movie_id in liked_movie_ids]
        logging.debug(f"Validated liked movie IDs: {liked_movie_ids}")
        
        for movie_id in liked_movie_ids:
            if not Preference.query.filter_by(user_id=current_user.id, movie_id=movie_id).first():
                logging.debug(f"Saving preference: user_id={current_user.id}, movie_id={movie_id}")
                new_preference = Preference(user_id=current_user.id, movie_id=movie_id)
                db.session.add(new_preference)
        db.session.commit()
        
        recommendations = hybrid_recommendation(current_user.id)

    except Exception as e:
        logging.error(f"Error: {e}")
        recommendations = [(str(e), None)]
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/delete_preference/<int:movie_id>', methods=['POST'])
@login_required
def delete_preference(movie_id):
    try:
        preference = Preference.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        if preference:
            db.session.delete(preference)
            db.session.commit()
            flash('Preference deleted successfully.', 'success')
        else:
            flash('Preference not found.', 'danger')
    except Exception as e:
        logging.error(f"Error deleting preference: {e}")
        flash('An error occurred while deleting the preference.', 'danger')
    
    return redirect(url_for('profile'))

@app.route('/search_movies')
def search_movies():
    query = request.args.get('q', '')
    if query:
        filtered_movies = df_movies[df_movies['title'].str.contains(query, case=False, na=False)]
        results = [{'id': int(movieId), 'text': title} for movieId, title in zip(filtered_movies['movieId'], filtered_movies['title'])]
    else:
        results = []
    return jsonify(results)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username and/or password.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already taken. Please choose a different one.', 'danger')
            return render_template('signup.html')
        
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    liked_movies = Preference.query.filter_by(user_id=current_user.id).all()
    liked_movie_ids = [movie.movie_id for movie in liked_movies]
    logging.debug(f"Retrieved liked_movie_ids for user_id={current_user.id}: {liked_movie_ids}")
    
    liked_movie_titles = df_movies[df_movies['movieId'].isin(liked_movie_ids)]['title'].tolist()
    logging.debug(f"Retrieved liked_movie_titles: {liked_movie_titles}")
    
    return render_template('profile.html', liked_movies=liked_movie_titles, movie_ids=liked_movie_ids)

if __name__ == '__main__':
    app.run(debug=False)
