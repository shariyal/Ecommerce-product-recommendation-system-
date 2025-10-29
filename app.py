from flask import Flask, request, render_template, jsonify
try:
    from flask_cors import CORS  # optional, for frontend integration
    _CORS_AVAILABLE = True
except Exception:
    _CORS_AVAILABLE = False
import pandas as pd
import random
_SQLALCHEMY_AVAILABLE = False  # SQLAlchemy not required for prototype
import sqlite3
import os
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommendation_code import (
    collaborative_filtering_recommendations,
    hybrid_recommendations,
    RecommendationExplainer,
)
from difflib import get_close_matches

# Serve templates and static directly from project root for this prototype
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
if _CORS_AVAILABLE:
    # Enable CORS for all routes in dev; tighten in production as needed
    CORS(app)

"""Utility to read CSV from multiple candidate paths"""
def _read_csv_candidates(cands: List[str]) -> pd.DataFrame:
    for p in cands:
        try:
            return pd.read_csv(p)
        except Exception:
            continue
    # Last resort: empty frame
    return pd.DataFrame()

# load files===========================================================================================================
# Keep CSVs as a fallback for UI rendering; API will prefer SQLite if available
trending_products = _read_csv_candidates(["models/trending_products.csv", "trending_products.csv"]) 
train_data = _read_csv_candidates(["models/clean_data.csv", "clean_data.csv"]) 

# SQLite configuration for prototype database
SQLITE_DB_PATH = os.environ.get("RECSYS_SQLITE", "recommender.db")

def get_sqlite_connection():
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print("SQLite connect error:", type(e).__name__, str(e))
        return None

# Ensure basic auth tables exist in SQLite (prototype)
def ensure_auth_tables():
    conn = get_sqlite_connection()
    if conn is None:
        return
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                password TEXT NOT NULL,
                ts TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                ts TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                product_id TEXT,
                event TEXT,
                rating REAL,
                ts TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
    except Exception as e:
        print("SQLite auth table init error:", type(e).__name__, str(e))
    finally:
        try:
            conn.close()
        except:
            pass

# database configuration---------------------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
# Initialize auth tables in SQLite
ensure_auth_tables()


# Removed SQLAlchemy models; using direct SQLite tables instead.

# Initialize multi-provider LLM explainer (Gemini/OpenAI/Ollama) with sqlite cache
explainer = RecommendationExplainer(db_path=SQLITE_DB_PATH)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def _ensure_tags_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'Tags' column exists by composing available text fields if missing."""
    if 'Tags' not in df.columns:
        # Build tags from available descriptive columns
        for col in ['Category', 'Brand', 'Name', 'Description']:
            if col not in df.columns:
                df[col] = ''
        df['Tags'] = (
            df['Category'].fillna('').astype(str) + ', '
            + df['Brand'].fillna('').astype(str) + ', '
            + df['Name'].fillna('').astype(str) + ', '
            + df['Description'].fillna('').astype(str)
        ).str.lower()
    else:
        # Normalize existing tags to string and lowercase
        df['Tags'] = df['Tags'].fillna('').astype(str).str.lower()
    return df


def content_based_recommendations(train_data, item_name, top_n=10):
    # Work on a copy to avoid mutating caller
    df = train_data.copy()
    # Ensure Tags exists
    df = _ensure_tags_column(df)

    # Validate product name, with case-insensitive fuzzy fallback
    all_names_series = df['Name'].dropna().astype(str) if 'Name' in df.columns else pd.Series([], dtype=str)
    name_map = {n.lower(): n for n in all_names_series}
    q = str(item_name).lower()
    if q not in name_map:
        suggestions = get_close_matches(q, list(name_map.keys()), n=1, cutoff=0.5)
        if not suggestions:
            print(f"Item '{item_name}' not found and no close match available.")
            return pd.DataFrame()
        item_name = name_map[suggestions[0]]
    else:
        item_name = name_map[q]

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = df[df['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Ensure expected columns exist
    for col in ['ReviewCount', 'Brand', 'ImageURL', 'Rating']:
        if col not in df.columns:
            df[col] = None

    # Get the details of the top similar items
    recommended_items_details = df.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details

# Utility to load products from SQLite (fallback to in-memory DataFrame)
def load_products_df(limit: Optional[int] = None) -> pd.DataFrame:
    conn = get_sqlite_connection()
    if conn is None:
        return train_data.copy() if limit is None else train_data.head(limit).copy()
    try:
        q = "SELECT * FROM products" + (" LIMIT ?" if limit else "")
        df = pd.read_sql_query(q, conn, params=(limit,) if limit else None)
        return df
    except Exception as e:
        print("SQLite load products error:", type(e).__name__, str(e))
        return train_data.copy() if limit is None else train_data.head(limit).copy()
    finally:
        try:
            conn.close()
        except:
            pass
# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "img_1.png",
    "img_2.png",
    "img_3.png",
    "img_4.png",
    "img_5.png",
    "img_6.png",
    "img_7.png",
    "img_8.png",
]


@app.route("/")
def index():
    # Prefer DB-backed products for homepage if available
    try:
        products_df = load_products_df(limit=8)
    except Exception:
        products_df = trending_products.head(8)
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(products_df))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=products_df,truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route("/main")
def main():
    # Provide an empty DataFrame so Jinja conditions won't error on first load
    empty_df = pd.DataFrame()
    return render_template('main.html', content_based_rec=empty_df)

# routes
@app.route("/index")
def indexredirect():
    products_df = load_products_df(limit=8)
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(products_df))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=products_df, truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = get_sqlite_connection()
        if conn is not None:
            try:
                conn.execute(
                    "INSERT INTO signup (username, email, password) VALUES (?, ?, ?)",
                    (username, email, password),
                )
                conn.commit()
            except Exception as e:
                print("Signup insert error:", type(e).__name__, str(e))
            finally:
                try:
                    conn.close()
                except:
                    pass

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        conn = get_sqlite_connection()
        if conn is not None:
            try:
                conn.execute(
                    "INSERT INTO signin (username, password) VALUES (?, ?)",
                    (username, password),
                )
                conn.commit()
            except Exception as e:
                print("Signin insert error:", type(e).__name__, str(e))
            finally:
                try:
                    conn.close()
                except:
                    pass

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed in successfully!'
                               )
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        # be robust if nbr is missing or invalid
        try:
            nbr = int(request.form.get('nbr', 10))
        except Exception:
            nbr = 10
        # Use DB-backed products for recommendations if available
        products_df = load_products_df()
        # Decide which dataset to use: prefer DB if it likely contains the product; otherwise use full train_data
        try:
            db_names = products_df['Name'].dropna().astype(str).tolist() if 'Name' in products_df.columns else []
            has_in_db = False
            if db_names:
                db_name_map = {n.lower(): n for n in db_names}
                q = str(prod).lower()
                if q in db_name_map:
                    has_in_db = True
                else:
                    # quick fuzzy check in DB names
                    has_in_db = bool(get_close_matches(q, list(db_name_map.keys()), n=1, cutoff=0.6))
            df_for_cb = products_df if has_in_db else train_data
        except Exception:
            df_for_cb = train_data

        content_based_rec = content_based_recommendations(df_for_cb, prod, top_n=nbr)

        if content_based_rec is None or content_based_rec.empty:
            # Build suggestions from a union of available names
            try:
                all_names = pd.concat([
                    (df_for_cb['Name'] if 'Name' in df_for_cb.columns else pd.Series(dtype=str)),
                    (train_data['Name'] if 'Name' in train_data.columns else pd.Series(dtype=str))
                ], ignore_index=True).dropna().astype(str).unique().tolist()
            except Exception:
                all_names = []
            q = str(prod).lower() if prod else ''
            name_map = {n.lower(): n for n in all_names}
            suggestions = get_close_matches(q, list(name_map.keys()), n=5, cutoff=0.5) if q else []
            display_suggestions = [name_map[s] for s in suggestions]
            message = "No recommendations available for this product."
            empty_df = pd.DataFrame()
            return render_template('main.html', message=message, content_based_rec=empty_df, suggestions=display_suggestions)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
            print(content_based_rec)
            print(random_product_image_urls)

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))


# ================== API endpoints for DB-backed prototype ==================
@app.route('/api/products', methods=['GET'])
def api_products():
    limit = request.args.get('limit', default=20, type=int)
    df = load_products_df(limit=limit)
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/interactions', methods=['POST'])
def api_interactions():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get('user_id', ''))
    product_id = str(data.get('product_id', ''))
    event = str(data.get('event', ''))
    rating = data.get('rating', None)
    if not (user_id and product_id and event):
        return jsonify({"error": "user_id, product_id, and event are required"}), 400
    conn = get_sqlite_connection()
    if conn is None:
        return jsonify({"error": "SQLite not available"}), 500
    try:
        conn.execute(
            "INSERT INTO interactions (user_id, product_id, event, rating) VALUES (?, ?, ?, ?)",
            (user_id, product_id, event, float(rating) if rating is not None else None),
        )
        conn.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": f"DB insert failed: {type(e).__name__}: {e}"}), 500
    finally:
        try:
            conn.close()
        except:
            pass

# -------------------------- REST API (v1) endpoints --------------------------
@app.route('/api/v1/health', methods=['GET'])
def api_v1_health():
    """Health check endpoint"""
    try:
        products_loaded = int(len(train_data)) if isinstance(train_data, pd.DataFrame) else 0
    except Exception:
        products_loaded = 0
    return jsonify({
        'status': 'healthy',
        'llm_available': bool(getattr(explainer, 'provider', None)),
        'llm_provider': getattr(explainer, 'provider', None) or 'none',
        'products_loaded': products_loaded,
    })


@app.route('/api/v1/recommendations/content-based', methods=['POST'])
def api_content_based():
    """
    Content-based recommendations endpoint
    """
    try:
        data = request.get_json()
        product_name = data.get('product_name')
        top_n = data.get('top_n', 10)
        include_explanations = data.get('include_explanations', False)
        
        if not product_name:
            return jsonify({'error': 'product_name is required'}), 400
        
        # Fuzzy match if not found
        all_names = train_data['Name'].dropna().unique().tolist()
        if product_name not in all_names:
            # Try close matches
            suggestions = get_close_matches(product_name, all_names, n=5, cutoff=0.6)
            if suggestions:
                # Use best match for recommendations
                best_match = suggestions[0]
                recs = content_based_recommendations(train_data, best_match, top_n=top_n)
                msg = f'Product "{product_name}" not found. Showing recommendations for closest match: "{best_match}".'
            else:
                return jsonify({
                    'message': f'No recommendations found for "{product_name}"',
                    'suggestions': suggestions,
                    'recommendations': []
                })
        else:
            recs = content_based_recommendations(train_data, product_name, top_n=top_n)
            msg = None
        
        if recs.empty:
            return jsonify({
                'message': f'No recommendations found for "{product_name}"',
                'recommendations': []
            })
        
        recommendations = recs.to_dict(orient='records')
        
        if include_explanations:
            for rec in recommendations:
                explanation = explainer.explain(
                    user_id=0,
                    product_name=rec['Name'],
                    features=f"similar to {product_name}",
                    user_history=''
                )
                rec['explanation'] = explanation
        
        response = {
            'query': product_name,
            'count': len(recommendations),
            'recommendations': recommendations
        }
        if msg:
            response['message'] = msg
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/recommendations/collaborative', methods=['POST'])
def api_v1_collaborative():
    """Collaborative filtering recommendations endpoint"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        top_n = int(data.get('top_n', 10))
        include_explanations = bool(data.get('include_explanations', False))
        if user_id is None:
            return jsonify({'error': 'user_id is required'}), 400

        # Build user history summary for explanations
        try:
            hist = train_data[train_data['ID'] == int(user_id)] if 'ID' in train_data.columns else pd.DataFrame()
            history_summary = ', '.join(hist['Name'].head(3).tolist()) if not hist.empty else 'Limited history'
        except Exception:
            history_summary = 'Limited history'

        recs = collaborative_filtering_recommendations(train_data, int(user_id), top_n=top_n)
        if recs is None or recs.empty:
            return jsonify({'message': f'No recommendations found for user {user_id}', 'recommendations': []})
        items = recs.to_dict(orient='records')
        if include_explanations:
            for rec in items:
                try:
                    rec['explanation'] = explainer.explain(
                        user_id=int(user_id),
                        product_name=rec.get('Name', ''),
                        features='based on similar users',
                        user_history=history_summary,
                    )
                except Exception:
                    rec['explanation'] = 'Explanation unavailable'
        return jsonify({'user_id': user_id, 'count': len(items), 'recommendations': items})
    except Exception as e:
        return jsonify({'error': f'{type(e).__name__}: {e}'}), 500


@app.route('/api/v1/recommendations/hybrid', methods=['POST'])
def api_v1_hybrid():
    """Hybrid recommendations endpoint"""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        product_name = data.get('product_name')
        top_n = int(data.get('top_n', 10))
        include_explanations = bool(data.get('include_explanations', False))
        if user_id is None or not product_name:
            return jsonify({'error': 'user_id and product_name are required'}), 400

        try:
            hist = train_data[train_data['ID'] == int(user_id)] if 'ID' in train_data.columns else pd.DataFrame()
            history_summary = ', '.join(hist['Name'].head(3).tolist()) if not hist.empty else 'Limited history'
        except Exception:
            history_summary = 'Limited history'

        recs = hybrid_recommendations(train_data, int(user_id), product_name, top_n=top_n)
        if recs is None or recs.empty:
            return jsonify({'message': 'No recommendations found', 'recommendations': []})
        items = recs.to_dict(orient='records')
        if include_explanations:
            for rec in items:
                try:
                    rec['explanation'] = explainer.explain(
                        user_id=int(user_id),
                        product_name=rec.get('Name', ''),
                        features=f'similar to {product_name} and user preferences',
                        user_history=history_summary,
                    )
                except Exception:
                    rec['explanation'] = 'Explanation unavailable'
        return jsonify({'user_id': user_id, 'query_product': product_name, 'count': len(items), 'recommendations': items})
    except Exception as e:
        return jsonify({'error': f'{type(e).__name__}: {e}'}), 500


@app.route('/api/v1/products/search', methods=['GET'])
def api_v1_search_products():
    """Search products by name"""
    query = (request.args.get('q') or '').lower()
    limit = int(request.args.get('limit', 20))
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    try:
        matches = train_data[train_data['Name'].str.lower().str.contains(query, na=False)]
        cols = [c for c in ['Name', 'Brand', 'Rating', 'ReviewCount', 'ImageURL'] if c in matches.columns]
        results = matches.head(limit)[cols].to_dict(orient='records')
        return jsonify({'query': query, 'count': len(results), 'products': results})
    except Exception as e:
        return jsonify({'error': f'{type(e).__name__}: {e}'}), 500


@app.route('/api/v1/user/<int:user_id>/interactions', methods=['POST'])
def api_v1_log_user_interaction(user_id: int):
    """Log user interaction into SQLite interactions table (prototype)"""
    try:
        data = request.get_json(silent=True) or {}
        product_id = str(data.get('product_id', ''))
        interaction_type = str(data.get('type', 'view'))
        rating = data.get('rating', None)
        if not product_id:
            return jsonify({'error': 'product_id is required'}), 400
        conn = get_sqlite_connection()
        if conn is None:
            return jsonify({'error': 'SQLite not available'}), 500
        conn.execute(
            "INSERT INTO interactions (user_id, product_id, event, rating) VALUES (?, ?, ?, ?)",
            (str(user_id), product_id, interaction_type, float(rating) if rating is not None else None),
        )
        conn.commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': f'{type(e).__name__}: {e}'}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__=='__main__':
    app.run(debug=True)