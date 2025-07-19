import datetime
from datetime import date, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from newsapi.newsapi_client import NewsApiClient
import psx
import requests
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend for Flask
import io
from io import BytesIO
import base64
import json
import os
from flask import send_from_directory
import plotly.graph_objects as go
import plotly

app = Flask(__name__)
app.secret_key = 'prediCt0r'  # Required for flashing messages

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Pakistan47@ug!'
app.config['MYSQL_DB'] = 'predictor_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize MySQL with error handling
try:
    mysql = MySQL(app)
    # Test the connection
    with app.app_context():
        mysql.connection.cursor()
except Exception as e:
    print(f"Error connecting to MySQL: {e}")
    print("Please make sure:")
    print("1. XAMPP is running")
    print("2. MySQL service is started in XAMPP")
    print("3. The database 'predictor_db' exists")
    print("4. The root user has the updated password")
    raise

# Ensure the data directory exists
DATA_DIR = os.path.join(app.root_path, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Fetch News
def fetch_news():
    newsapi = NewsApiClient(api_key="c842bd75db8240d0a9cf69d6088505f3")
    headlines = newsapi.get_everything(
        q="Pakistan Stock OR Pakistan Stock Exchange OR PSX OR Pakistan Economy OR Pakistan Finance",
        language="en",
        sort_by="relevancy"
    )
    articles = headlines["articles"]
    
    # Process the articles as before
    news_data = []
    for article in articles:
        title = article["title"]
        description = article["description"]
        sentiment = TextBlob(title + " " + (description or "")).sentiment.polarity
        impact = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        news_data.append({"title": title, "impact": impact, "description": description})
    return news_data

# Fetch Indices Data
def fetch_indices():
    top_100_tickers = [
        "WTL", "CNERGY", "BOP", "HBL", "PSO", "UBL", "SHEL", "LUCK", "PAEL",
        "MLCF", "PIBTL", "KEL", "SSGC", "FCCL", "PRL", "HUBC", "FFL", "DGKC",
        "OGDC", "POWER", "HASCOL", "PREMA", "TPLP", "PPL", "NBP", "SEARL", "BAFL", "PTC",
        "TREET"
    ]

    indices_data = []
    
    for ticker in top_100_tickers:  # No slicing here to fetch all indices
        try:
            # Fetch stock data for the current symbol
            stock_data = psx.stocks(ticker, start=date.today() - timedelta(days=1), end=date.today())
            for record in stock_data.to_dict(orient="records"):
                record['ticker'] = ticker  # Add ticker name to each record
                indices_data.append(record)  # Append the record to the list
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return indices_data

def fetch_stock_data(ticker):
    try:
        start_date = date(2015, 1, 1)
        end_date = date.today() - timedelta(days=1)

        # Fetch stock data using the `stocks` function from the psx module
        stock_data = psx.stocks(ticker, start=start_date, end=end_date)
        stock_data.columns = stock_data.columns.str.strip()

        if 'date' not in stock_data.columns:
            stock_data['date'] = stock_data.index
        
        stock_data['date'] = pd.to_datetime(stock_data['date'])

        return stock_data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def fetch_news_for_ticker(ticker):
    newsapi = NewsApiClient(api_key="c842bd75db8240d0a9cf69d6088505f3")
    try:
        # Search for news articles related to the given ticker
        headlines = newsapi.get_everything(
            q=ticker,  # Searching for the ticker symbol in news articles
            language="en",
            sort_by="relevancy"
        )
        articles = headlines["articles"]
        # Parse the articles to extract relevant information
        news_data = []
        for article in articles:
            title = article["title"]
            description = article["description"]
            publishedAt = article["publishedAt"]
            sentiment = TextBlob(title + " " + (description or "")).sentiment.polarity
            impact = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            news_data.append({"title": title, "impact": impact, "description": description,"publishedAt":publishedAt})

        return news_data
    except Exception as e:
        print(f"Error fetching news for ticker {ticker}: {e}")
        return []  # Return an empty list if an error occurs

# Helper function to compute Simple Moving Average (SMA)
def compute_sma(data, window=5):
    return data['Close'].rolling(window=window).mean()

# Helper function to compute Exponential Moving Average (EMA)
def compute_ema(data, window=5):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Helper function to compute Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Helper function to compute Sentiment score
def compute_sentiment(news_data):
    sentiments = []
    published_dates = []

    for article in news_data:
        # Compute sentiment for each article's title and description
        title = article.get('title', '')
        description = article.get('description', '')
        sentiment = TextBlob(title + " " + description).sentiment.polarity
        sentiments.append(sentiment)

        # Access the 'publishedAt' key using get() to avoid KeyError
        published_at = article.get('publishedAt', None)  # Default to None if not found

        if published_at:
            published_dates.append(pd.to_datetime(published_at))
        else:
            published_dates.append(None)

    # Filter out articles that don't have a valid date (None values)
    valid_sentiments = [s for s, d in zip(sentiments, published_dates) if d is not None]
    valid_dates = [d for d in published_dates if d is not None]

    # Create the sentiment series with valid sentiments and dates
    sentiment_series = pd.Series(valid_sentiments, index=pd.to_datetime(valid_dates))

    # Resample sentiment by day (average sentiment per day)
    sentiment_daily = sentiment_series.resample('D').mean()
    print(sentiment_daily)
    return sentiment_daily

# Function to combine stock data and sentiment data
def combine_data_with_sentiment(stock_data, news_data, sentiment_window=5):
    # Compute technical indicators
    stock_data['SMA_5'] = compute_sma(stock_data, window=5)
    stock_data['EMA_5'] = compute_ema(stock_data, window=5)
    stock_data['RSI'] = compute_rsi(stock_data)
    # Handle missing values in stock data
    stock_data = stock_data.ffill()  # Forward fill missing values
    stock_data.index = stock_data.index.tz_convert('UTC') if stock_data.index.tz else stock_data.index.tz_localize('UTC')

    # Compute sentiment for the news (assuming sentiment is computed per day)
    sentiment = compute_sentiment(news_data)
    sentiment.index = sentiment.index.tz_convert('UTC') if sentiment.index.tz else sentiment.index.tz_localize('UTC')

    # If sentiment data is empty, return the stock data with technical indicators only
    if sentiment.empty:
        print("Sentiment data is empty. Proceeding with stock data only.")
        stock_data['Sentiment'] = np.nan  # No sentiment to use, set to NaN
        return stock_data[['SMA_5', 'EMA_5', 'RSI', 'Close', 'Volume']]  # Return without sentiment

    # Align sentiment with stock data by the date index
    sentiment = sentiment.reindex(stock_data.index, method='ffill')  # Forward fill sentiment for missing dates

    # Ensure sentiment is added to the stock data
    stock_data['Sentiment'] = sentiment

    # Feature columns with sentiment
    return stock_data[['SMA_5', 'EMA_5', 'RSI', 'Sentiment', 'Close', 'Volume']]

# Function to train model
def train_model(data):
    # Prepare features and target variable
    feature_columns = ['SMA_5', 'EMA_5', 'RSI', 'Volume']
    if 'Sentiment' in data.columns:
        feature_columns.append('Sentiment')  # Add 'Sentiment' only if it exists

    X = data[feature_columns]
    y = data['Close'].shift(-1)  # Predicting the next day's close price

    # Drop the last row since it doesn't have a target value
    X = X[:-1]
    y = y.dropna()

    # Ensure enough data is available
    if len(X) < 10:  # You can adjust this threshold based on your needs
        print(f"Error: Not enough data to train the model. Data length: {len(X)}")
        return None  # Return None if not enough data

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: No data for training.")
        return None  # Return None if no training data

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# Function to generate price difference graph
def create_price_diff_graph(stock_data):
    try:
        # Ensure stock_data contains necessary columns: 'Date', 'Close', 'Open'
        fig, ax = plt.subplots()

        # Plotting price difference (close - open)
        ax.plot(stock_data['date'], stock_data['Close'] - stock_data['Open'], label='Price Difference')
        ax.set_title("Price Difference")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price Difference")
        ax.legend()

        return fig  # Ensure it returns a matplotlib figure
    except Exception as e:
        print(f"Error in create_price_diff_graph: {e}")
        return None

def create_performance_graph(stock_data):
    try:
        # Ensure stock_data contains necessary columns: 'Date', 'Close'
        fig, ax = plt.subplots()

        ax.plot(stock_data['date'], stock_data['Close'], label='Performance')
        ax.set_title("Stock Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        return fig  # Ensure it returns a matplotlib figure
    except Exception as e:
        print(f"Error in create_performance_graph: {e}")
        return None

def create_prediction_graph(stock_data, model):
    try:
        fig, ax = plt.subplots()
        features = stock_data[['SMA_5', 'EMA_5', 'RSI', 'Volume','Sentiment']].dropna()
        print("Feature = ",features)
        if model is None:
            print("Model is None")
        # Assume prediction is based on the model's predictions on stock data
        # (Adjust this part according to your model and prediction logic)
        ax.plot(features.index, model.predict(features), label="Prediction")
        
        ax.set_title("Prediction Graph")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Value")
        ax.legend()

        print("Prediction graph generated successfully!")  # Debug line
        return fig
    except Exception as e:
        print(f"Error in create_prediction_graph: {e}")
        return None

# Convert plots to base64 so they can be displayed in the web page
def plot_to_base64(fig):
    if fig is None:
        return None

    try:
        # Save the figure to a BytesIO buffer
        img = io.BytesIO()
        fig.savefig(img, format='png')  # Saving the figure as PNG
        img.seek(0)  # Rewind the buffer to the beginning

        # Encode the image as base64
        img_base64 = base64.b64encode(img.read()).decode('utf-8')

        return img_base64  # Return the base64 string
    except Exception as e:
        print(f"Error in plot_to_base64: {e}")
        return None

@app.route('/')
def home():
    return render_template('login.html')

# Route to fetch and store data
@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    if 'logged_in' not in session:
        return jsonify({'error': 'Not authorized'}), 401

    try:
        
        # Store in JSON file
        data_path = os.path.join(DATA_DIR, 'psx_data.json')
        # check if the file already exists
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                indices_data = json.load(f)
            return jsonify({
                'success': True,
                'message': 'Data already exists',
                'data': indices_data[:10],  # Return top 10 for immediate display
                'count': len(indices_data)
            })
        # Fetch fresh data
        indices_data = fetch_indices()
        with open(data_path, 'w') as f:
            json.dump(indices_data, f)
        
        return jsonify({
            'success': True,
            'message': 'Data fetched and stored successfully',
            'data': indices_data[:10],  # Return top 10 for immediate display
            'count': len(indices_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to serve the JSON file
@app.route('/data/<filename>')
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)

# Index Page Route
@app.route('/index')
def index():
    if 'logged_in' not in session:
        flash('You need to log in first!', 'danger')
        return redirect(url_for('home'))

    # Try to load cached data
    data_path = os.path.join(DATA_DIR, 'psx_data.json')
    indices = []
    
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            try:
                indices = json.load(f)
            except json.JSONDecodeError:
                indices = fetch_indices()
    else:
        indices = fetch_indices()

    news = fetch_news()
    top_10_indices = indices[:10]
    return render_template("index.html", 
                         news=news, 
                         top_10=top_10_indices, 
                         all_indices=indices,
                         now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Predictions Page Route
@app.route('/predict/<ticker>', methods=['GET'])
# def predict(ticker):
#     # Fetch stock data and news for the specific ticker
#     stock_data = fetch_stock_data(ticker)
#     news_data = fetch_news_for_ticker(ticker)

#     # If no stock data or news data, return an error page
#     if stock_data.empty or not news_data:
#         flash("Insufficient data to make a prediction.", "danger")
#         return redirect(url_for('index'))  # Redirect back to index if no data

#     # Combine stock data and sentiment data
#     combined_data = combine_data_with_sentiment(stock_data, news_data)
#     # Check if sentiment data is available
#     if 'Sentiment' not in combined_data.columns or combined_data['Sentiment'].isnull().all():
#         flash("Sentiment data is missing. Prediction will be made using stock data only.", "warning")

#         # Use only stock data columns if sentiment data is missing
#         combined_data = combined_data[['SMA_5', 'EMA_5', 'RSI', 'Close', 'Volume']]
        
#     # Train the model (or load pre-trained model)
#     model = train_model(combined_data)

#     # If model training failed, return an error message
#     if model is None:
#         print("Mode is none ")
#         flash("Error: Not enough data to train the model.", "danger")
#         return redirect(url_for('index'))  # Redirect back to index if model training fails

#     # Generate all required graphs: price difference, performance, and prediction
#     price_diff_graph = create_price_diff_graph(stock_data)
#     performance_graph = create_performance_graph(stock_data)

#     # If sentiment data is available, generate the prediction graph
#     prediction_graph = None
#     if 'Sentiment' in combined_data.columns and not combined_data['Sentiment'].isnull().all():
#         prediction_graph = create_prediction_graph(combined_data, model)

#     # Check if any graph generation failed
#     if price_diff_graph is None or performance_graph is None or (prediction_graph is None and 'Sentiment' in combined_data.columns):
#         flash("Error generating prediction graphs.", "danger")
#         return redirect(url_for('index'))  # Redirect back to index if graph generation fails

#     # Convert the graphs to base64
#     price_diff_image = plot_to_base64(price_diff_graph)
#     performance_image = plot_to_base64(performance_graph)
#     prediction_image = plot_to_base64(prediction_graph) if prediction_graph else None

#     # Render the template with the images and ticker information
#     return render_template('predict.html', 
#                          ticker=ticker, SELECT id, username, email, password FROM users WHERE username = %s;

#                          price_diff_image=price_diff_image, 
#                          performance_image=performance_image, 
#                          prediction_image=prediction_image)

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        is_admin = request.form.get('is_admin', 'No')  # Default to 'No' if not provided

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create cursor
        cur = mysql.connection.cursor()

        # Check if username or email already exists
        cur.execute("SELECT * FROM users WHERE username=%s OR email=%s", (username, email))
        user = cur.fetchone()

        if user:
            flash('Username or Email already exists!', 'danger')
            return redirect(url_for('register'))

        # Insert new user into database
        cur.execute("INSERT INTO users(username, email, password, is_admin) VALUES(%s, %s, %s, %s)",
                    (username, email, hashed_password, is_admin))

        # Commit and close connection
        mysql.connection.commit()
        cur.close()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('home'))

    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Fetch user details from the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, email, password, is_admin FROM users WHERE username=%s", [username])
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[3], password):  # Assuming password is hashed
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['is_admin'] = user[4]  # Store is_admin in session
            
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/not_admin')
def not_admin():
    return render_template('not_admin.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    # Check if user is logged in and is an admin
    if 'logged_in' not in session or session.get('is_admin') != 'Yes':
        flash('You are not authorized to access this page.', 'danger')
        return redirect(url_for('not_admin'))  # Redirect to the homepage or a different page

    # Fetch users or other admin data to display on the dashboard
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, username, email FROM users")
    users = cur.fetchall()
    cur.close()

    return render_template('admin_dashboard.html', users=users)

# Edit User Route
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    # Check if user is logged in and is an admin
    if 'logged_in' not in session or not session.get('is_admin', False):
        flash('You are not authorized to access this page.', 'danger')
        return redirect(url_for('index'))

    cur = mysql.connection.cursor()
    
    # If the method is GET, fetch the user's current details
    if request.method == 'GET':
        cur.execute("SELECT id, username, email, password FROM users WHERE id=%s", [user_id])
        user = cur.fetchone()
        cur.close()

        if user:
            return render_template('edit_user.html', user=user)
        else:
            flash('User not found.', 'danger')
            return redirect(url_for('admin_dashboard'))
    
    # If the method is POST, update the user details
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Fetch the current password from the database to avoid overwriting it if unchanged
        cur.execute("SELECT password FROM users WHERE id=%s", [user_id])
        current_password = cur.fetchone()[0]

        # Optionally hash password if it was changed
        if password:
            hashed_password = generate_password_hash(password)
        else:
            # Retain the existing password if no new password is provided
            hashed_password = current_password
        
        cur.execute("""
            UPDATE users 
            SET username=%s, email=%s, password=%s 
            WHERE id=%s
        """, (username, email, hashed_password, user_id))
        mysql.connection.commit()
        cur.close()

        flash('User details updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

# Delete User Route
@app.route('/delete_user/<int:user_id>', methods=['GET'])
def delete_user(user_id):
    # Check if user is logged in and is an admin
    if 'logged_in' not in session or not session.get('is_admin', False):
        flash('You are not authorized to access this page.', 'danger')
        return redirect(url_for('index'))

    # Create cursor
    cur = mysql.connection.cursor()

    # Delete the user from the database
    cur.execute("DELETE FROM users WHERE id=%s", [user_id])
    mysql.connection.commit()
    cur.close()

    flash('User deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

# Logout Route   
@app.route('/logout')
def logout():
    session.clear()  # This will remove all session data
    flash('You have been logged out!', 'success')
    return redirect(url_for('home'))




# --- Data Fetching and Processing Functions ---
def fetch_stock_data(ticker):
    # Replace with actual logic to fetch stock data (e.g., using Alpha Vantage or Yahoo Finance)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    data = pd.DataFrame({
        'Close': [100 + i * 0.5 for i in range(30)],
        'RSI': [50 + i * 0.3 for i in range(30)],
        'SMA_5': [100 + i * 0.4 for i in range(30)],
        'EMA_5': [100 + i * 0.45 for i in range(30)],
    }, index=dates)
    return data

def fetch_news_for_ticker(ticker):
    # Replace with actual logic to fetch news for the ticker
    return [{'date': pd.Timestamp.today(), 'sentiment': 0.5}]

def combine_data_with_sentiment(stock_data, news_data):
    # Combine stock data with sentiment data
    sentiment_series = compute_sentiment(news_data)
    stock_data['Sentiment'] = sentiment_series
    return stock_data

def train_model(data):
    # Placeholder function to train a predictive model
    return None  # Replace with actual model training logic

def compute_sentiment(news_data):
    # Calculate sentiment based on news data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    return pd.Series([0.1 * (i % 5 - 2) for i in range(30)], index=dates)

# --- Plotly Graph Functions ---
def create_plotly_graph(x, y, title, x_label, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')  # Replace with actual homepage template

@app.route('/predict/<ticker>', methods=['GET']) 
def predict(ticker):
    stock_data = fetch_stock_data(ticker)
    news_data = fetch_news_for_ticker(ticker)

    if stock_data.empty or not news_data:
        flash("Insufficient data to make a prediction.", "danger")
        return redirect(url_for('index'))

    combined_data = combine_data_with_sentiment(stock_data, news_data)
    model = train_model(combined_data)

    sentiment_series = compute_sentiment(news_data)

    # Convert plots to JSON
    price_diff_json = create_plotly_graph(stock_data.index, stock_data['Close'], "Price Difference Over Time", "Date", "Price")
    performance_json = create_plotly_graph(stock_data.index, stock_data['Close'].rolling(5).mean(), "Performance (SMA)", "Date", "SMA")
    prediction_json = create_plotly_graph(stock_data.index, stock_data['Close'] * 1.03, "Prediction (Dummy)", "Date", "Predicted Price")
    sentiment_json = create_plotly_graph(sentiment_series.index, sentiment_series, "Sentiment Over Time", "Date", "Sentiment Score")

    latest_row = stock_data.iloc[-1]
    stock_info = {
        'current_price': round(latest_row['Close'], 2),
        'market_cap': "N/A",  # Replace with actual data if available
        'high_52w': round(stock_data['Close'].max(), 2),
        'low_52w': round(stock_data['Close'].min(), 2),
        'pe_ratio': "N/A",  # Replace with actual data if available
        'dividend_yield': "N/A"  # Replace with actual data if available
    }

    technical_indicators = [
        {'name': 'RSI (14)', 'value': round(combined_data['RSI'].iloc[-1], 2), 'status': 'Neutral'},
        {'name': 'SMA (5)', 'value': round(combined_data['SMA_5'].iloc[-1], 2), 'status': 'Neutral'},
        {'name': 'EMA (5)', 'value': round(combined_data['EMA_5'].iloc[-1], 2), 'status': 'Neutral'},
    ]

    risk_metrics = [
        {'name': 'Volatility', 'value': 'Moderate', 'level': '65%'},
        {'name': 'Beta', 'value': '0.85', 'level': '45%'},
        {'name': 'Sharpe Ratio', 'value': '1.2', 'level': '75%'}
    ]

    related_stocks = [
        {'name': 'DG Khan Cement', 'price': 125.50, 'change': 2.5},
        {'name': 'Fauji Cement', 'price': 95.75, 'change': -1.2},
        {'name': 'Maple Leaf Cement', 'price': 45.30, 'change': 0.8},
        {'name': 'Bestway Cement', 'price': 180.25, 'change': 1.5}
    ]

    prediction_stats = {
        'predicted_price': round(combined_data['Close'].iloc[-1] * 1.03, 2),
        'confidence_level': 85,
        'time_horizon': '30 days',
        'historical_accuracy': 82
    }

    return render_template(
        "predict.html",
        ticker=ticker,
        stock_data=stock_info,
        technical_indicators=technical_indicators,
        risk_metrics=risk_metrics,
        related_stocks=related_stocks,
        prediction_stats=prediction_stats,
        price_diff_json=price_diff_json,
        performance_json=performance_json,
        prediction_json=prediction_json,
        sentiment_json=sentiment_json
    )


if __name__ == "__main__":
    app.run(debug=True)