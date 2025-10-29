# Top-level demo code removed. Use `main()` and `run_examples(train_data)` to run demos.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Module-level placeholder so static analyzers (Pylance) don't report undefined-variable
# The real DataFrame is created by `main()` and assigned to this name at runtime.
train_data = None

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import os
from scipy.sparse import coo_matrix
import argparse
import glob
import sys
from pathlib import Path

# Helper: adaptable data loader with schema mapping
from typing import Dict, Optional
import io
import csv
import datetime

# Database: SQLAlchemy models for interactions (guarded import)
_SQLALCHEMY_AVAILABLE = True
try:
    from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
    from sqlalchemy.orm import declarative_base, sessionmaker

    Base = declarative_base()


    class UserInteraction(Base):
        __tablename__ = 'user_interactions'
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=True)
        product_id = Column(Integer, nullable=True)
        interaction_type = Column(String(32), nullable=False)  # view, purchase, etc.
        timestamp = Column(DateTime, default=datetime.datetime.utcnow)


    def init_db(db_path: str = 'recommender.db'):
        """Create SQLite engine and ensure tables exist."""
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        return engine


    def persist_products_sqlite(train_df: pd.DataFrame, engine) -> None:
        """Persist product metadata to the SQLite DB using pandas. Safe and idempotent."""
        try:
            train_df.to_sql('products', engine, if_exists='replace', index=False)
        except Exception:
            # fallback to file-based CSV if DB write fails
            train_df.to_csv('products_export.csv', index=False)


    def log_interaction(engine, user_id: int, product_id: int, interaction_type: str = 'view', ts: datetime.datetime | None = None):
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            ui = UserInteraction(user_id=int(user_id) if user_id is not None else None,
                                 product_id=int(product_id) if product_id is not None else None,
                                 interaction_type=str(interaction_type),
                                 timestamp=ts or datetime.datetime.utcnow())
            session.add(ui)
            session.commit()
        finally:
            session.close()
except Exception:
    _SQLALCHEMY_AVAILABLE = False

    def init_db(db_path: str = 'recommender.db'):
        print('SQLAlchemy not installed; init_db() is a no-op')
        return None


    def persist_products_sqlite(train_df: pd.DataFrame, engine) -> None:
        # fallback to CSV export when SQLAlchemy not available
        try:
            train_df.to_csv('products_export.csv', index=False)
            print('Products exported to products_export.csv (SQLAlchemy unavailable)')
        except Exception:
            pass


    def log_interaction(engine, user_id: int, product_id: int, interaction_type: str = 'view', ts: datetime.datetime | None = None):
        # Provide a sqlite3-backed fallback so interactions are recorded even when SQLAlchemy
        # isn't installed. The `engine` parameter is ignored in this fallback and `recommender.db`
        # is used (or the path passed to init_db earlier).
        try:
            import sqlite3 as _sqlite
            db_path = 'recommender.db'
            # try to extract a path if an engine-like string was passed
            try:
                if isinstance(engine, str) and engine:
                    db_path = engine
            except Exception:
                pass
            conn = _sqlite.connect(db_path)
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    product_id INTEGER,
                    interaction_type TEXT,
                    timestamp TEXT
                )
            ''')
            ts_val = (ts or datetime.datetime.utcnow()).isoformat()
            cur.execute('INSERT INTO user_interactions (user_id, product_id, interaction_type, timestamp) VALUES (?,?,?,?)',
                        (int(user_id) if user_id is not None else None,
                         int(product_id) if product_id is not None else None,
                         str(interaction_type), ts_val))
            conn.commit()
            conn.close()
            # Silent success for demo
        except Exception as _e:
            print('Fallback log_interaction failed:', type(_e).__name__, _e)


def get_llm_explanation(client_wrapper, user_id, product_name: str, reason_features: str) -> str:
    """Return an explanation using the unified LLM wrapper if available.

    Falls back to a short heuristic explanation if client is not configured.
    """
    try:
        if client_wrapper is not None:
            try:
                return explain_recommendation_unified(client_wrapper, user_id, product_name, reason_features)
            except Exception as inner:
                print('LLM unified explain failed:', type(inner).__name__, inner)
        # deterministic heuristic fallback
        pf = str(product_name) if product_name is not None else 'this product'
        rf = str(reason_features) if reason_features else 'similar features'
        return f"Recommended because {pf} matches {rf} and has positive signals from other shoppers."
    except Exception as e:
        print('Unexpected error building explanation:', type(e).__name__, e)
        return 'No explanation available.'


def _detect_sep(sample: str) -> str:
    # Try common delimiters
    for delim in ["\t", ",", ";", "|"]:
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample, delimiters=[delim])
            return dialect.delimiter
        except Exception:
            continue
    return "\t"  # sensible default for TSV


def load_and_clean_dataset(path: str, schema: Optional[Dict[str, str]] = None, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw e-commerce data and return a DataFrame with canonical columns:
        ID, ProdID, Rating, ReviewCount, Category, Brand, Name, ImageURL, Description, Tags

    Parameters
    ----------
    path : str
        File path to CSV/TSV.
    schema : Optional[Dict[str,str]]
        Mapping from logical keys to actual source column names.
        Supported keys: user_id, product_id, rating, review_count, category, brand,
                        name, image_url, description, tags
        If None, defaults for the Walmart sample are used.
    sep : Optional[str]
        Delimiter override. If None, attempts auto-detection.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized column names and types.
    """
    # Defaults tailored to the provided Walmart sample
    default_schema = {
        "user_id": "Uniq Id",
        "product_id": "Product Id",
        "rating": "Product Rating",
        "review_count": "Product Reviews Count",
        "category": "Product Category",
        "brand": "Product Brand",
        "name": "Product Name",
        "image_url": "Product Image Url",
        "description": "Product Description",
        "tags": "Product Tags",
    }

    schema = {**default_schema, **(schema or {})}

    # Auto-detect delimiter by sampling the first ~8KB if not provided
    if sep is None:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(8192)
        sep = _detect_sep(sample)

    # Read the file using detected or provided delimiter
    df = pd.read_csv(path, sep=sep)

    # Rename columns to canonical names based on schema
    rename_map = {
        schema["user_id"]: "ID",
        schema["product_id"]: "ProdID",
        schema["rating"]: "Rating",
        schema["review_count"]: "ReviewCount",
        schema["category"]: "Category",
        schema["brand"]: "Brand",
        schema["name"]: "Name",
        schema["image_url"]: "ImageURL",
        schema["description"]: "Description",
        schema["tags"]: "Tags",
    }

    # Keep only available columns among those requested
    available_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=available_map)

    # Ensure all expected columns exist (create missing with sensible defaults)
    expected_cols = [
        "ID",
        "ProdID",
        "Rating",
        "ReviewCount",
        "Category",
        "Brand",
        "Name",
        "ImageURL",
        "Description",
        "Tags",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan if col in ["Rating", "ReviewCount"] else ""

    # Type coercions and cleaning
    def _to_numeric(series: pd.Series) -> pd.Series:
        # Try direct to_numeric first
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().any():
            return coerced
        # Fallback: extract digits when values are mixed strings
        extracted = pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        return extracted

    df["ID"] = _to_numeric(df["ID"])  # robust user IDs
    df["ProdID"] = _to_numeric(df["ProdID"])  # robust product IDs
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["ReviewCount"] = pd.to_numeric(df["ReviewCount"], errors="coerce")

    # Normalize sentinel/invalid IDs commonly present in some exports
    # Replace sentinel -2147483648 with NaN and drop rows missing either ID or ProdID
    try:
        df['ID'] = df['ID'].replace({-2147483648: np.nan})
        df['ProdID'] = df['ProdID'].replace({-2147483648: np.nan})
    except Exception:
        pass

    # Fill missing values consistent with downstream cells
    df["Rating"] = df["Rating"].fillna(0.0)
    df["ReviewCount"] = df["ReviewCount"].fillna(0.0)
    for txt_col in ["Category", "Brand", "Description", "Tags", "Name", "ImageURL"]:
        df[txt_col] = df[txt_col].fillna("")

    # Final column subset and order
    df = df[expected_cols]
    # Drop rows missing critical IDs (after normalization)
    try:
        df = df.dropna(subset=['ID','ProdID']).reset_index(drop=True)
        df['ID'] = df['ID'].astype(np.int64)
        df['ProdID'] = df['ProdID'].astype(np.int64)
    except Exception:
        # If conversion fails, leave as-is but warn (handled elsewhere)
        pass
    return df

def _find_data_file(provided_path: str | None = None) -> str | None:
    """Return a path to a data file.

    If provided_path is given and exists, return it. Otherwise search the script
    directory for common filenames and extensions (.tsv, .csv). Returns None if
    nothing found.
    """
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return str(p)
        # allow relative to script dir
        p = Path(__file__).parent.joinpath(provided_path)
        if p.exists():
            return str(p)
        return None

    search_dir = Path(__file__).parent
    # preferred names first
    preferred = ["clean_data.csv", "trending_products.csv"]
    for name in preferred:
        p = search_dir.joinpath(name)
        if p.exists():
            return str(p)

    # then common extensions
    for ext in ("*.tsv", "*.csv"):
        found = list(search_dir.glob(ext))
        if found:
            return str(found[0])

    return None


def validate_data_structure(df: pd.DataFrame) -> None:
    """Prints helpful diagnostics about the loaded dataframe."""
    if df is None:
        print("❌ train_data is None")
        return
    print("Data Structure:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    try:
        print(f"  User ID range: {df['ID'].min()} to {df['ID'].max()}")
    except Exception:
        print("  User ID range: unavailable")
    try:
        print(f"  Product ID range: {df['ProdID'].min()} to {df['ProdID'].max()}")
    except Exception:
        print("  Product ID range: unavailable")
    try:
        print(f"  Rating range: {df['Rating'].min()} to {df['Rating'].max()}")
    except Exception:
        print("  Rating range: unavailable")

def enhanced_collaborative_filtering(train_data, target_user_id, top_n=10):
    """Enhanced collaborative filtering with better similarity handling"""
    try:
        # Build user-item matrix
        user_item_matrix = train_data.pivot_table(
            index='ID', columns='ProdID', values='Rating', aggfunc='mean'
        ).fillna(0.0)
        
        if target_user_id not in user_item_matrix.index:
            return pd.DataFrame()

        # Use adjusted ratings if original ratings are mostly zeros
        if (user_item_matrix.values.flatten() == 0).mean() > 0.8:
            print("Using popularity-adjusted ratings for CF")
            # Add small popularity bonus based on review count
            item_popularity = train_data.groupby('ProdID')['ReviewCount'].mean()
            for idx, user_id in enumerate(user_item_matrix.index):
                user_ratings = user_item_matrix.iloc[idx].values
                # Add small bonus for popular items
                adjusted = user_ratings + 0.1 * (item_popularity.reindex(user_item_matrix.columns).fillna(0).values / 100)
                user_item_matrix.iloc[idx] = np.clip(adjusted, 0, 5)
        
        user_similarity = cosine_similarity(user_item_matrix)
        np.fill_diagonal(user_similarity, 0.0)

        target_idx = user_item_matrix.index.get_loc(target_user_id)
        user_similarities = user_similarity[target_idx]

        # Filter out very low similarities
        if np.max(user_similarities) < 0.1:
            print("No strong user similarities found - using item popularity")
            item_means = train_data.groupby('ProdID')['Rating'].mean()
            predicted_scores = item_means.reindex(user_item_matrix.columns).fillna(0).values
        else:
            numerator = user_similarities @ user_item_matrix.values
            denominator = np.abs(user_similarities).sum()
            predicted_scores = numerator / denominator if denominator > 0 else np.zeros_like(numerator)

        already_rated = user_item_matrix.iloc[target_idx].values > 0
        predicted_scores_masked = np.where(already_rated, -np.inf, predicted_scores)

        best_indices = np.argsort(predicted_scores_masked)[-top_n:][::-1]
        recommended_prod_ids = user_item_matrix.columns.values[best_indices]

        rec_df = (train_data[train_data['ProdID'].isin(recommended_prod_ids)]
                  .drop_duplicates(subset=['ProdID'])
                  .set_index('ProdID')
                  .loc[recommended_prod_ids, ['Name','ReviewCount','Brand','ImageURL','Rating']]
                  .reset_index())
        rec_df.rename(columns={'ProdID':'Product'}, inplace=True)
        rec_df['PredScore'] = np.round(predicted_scores[best_indices], 3)
        
        return rec_df.head(top_n)
        
    except Exception as e:
        print(f"Enhanced CF error: {e}")
        return pd.DataFrame()

def create_demo_hybrid_recommendations(train_data):
    """Create working hybrid recommendations for demo"""
    try:
        # Get content-based recommendations
        sample_product = train_data['Name'].iloc[0]
        content_recs = content_based_recommendations(train_data, sample_product, top_n=5)
        
        # Get collaborative recommendations
        sample_user = int(train_data['ID'].iloc[0])
        collab_recs = enhanced_collaborative_filtering(train_data, sample_user, top_n=5)
        
        # Combine and add LLM-ready columns
        hybrid_df = pd.concat([content_recs, collab_recs]).drop_duplicates().head(8)
        
        # Add required columns for LLM explanations
        hybrid_df['UserID'] = sample_user
        hybrid_df['Category'] = 'Beauty'  # Sample category
        hybrid_df['Tags'] = 'cosmetics, beauty products'
        
        return hybrid_df
    except Exception as e:
        print(f"Demo hybrid creation failed: {e}")
        return pd.DataFrame()

def test_basic_recommendations(train_data: pd.DataFrame):
    """Quick smoke tests for content-based and collaborative recommenders."""
    if train_data is None or train_data.empty:
        print("No data available for recommendation tests")
        return None
    try:
        # Content-based test
        sample_products = train_data['Name'].dropna().unique()[:5]
        found_cb = False
        for product in sample_products:
            print(f"Testing content-based recommendations for: {str(product)[:60]}...")
            recs = content_based_recommendations(train_data, product, top_n=3)
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                print(f"✓ Content-based success: found {len(recs)} recs for sample product")
                found_cb = True
                break
            else:
                print("✗ Content-based: no recs for this product")

        # Collaborative test
        sample_users = train_data['ID'].dropna().unique()[:3]
        found_cf = False
        for user in sample_users:
            print(f"Testing collaborative filtering for user: {user}")
            recs = collaborative_filtering_recommendations(train_data, int(user), top_n=3)
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                print(f"✓ Collaborative success: found {len(recs)} recs for user {user}")
                found_cf = True
                break
            else:
                print("✗ Collaborative: no recs for this user")

        if not found_cb and not found_cf:
            print("Warning: both CB and CF tests failed. Check Tags and user-item density.")
        return True
    except Exception as e:
        print(f"Recommendation test failed: {type(e).__name__} {e}")
        return None


def safe_persist_products(train_data: pd.DataFrame, db_path: str = 'recommender.db') -> None:
    """Safely persist products into a SQLite database using sqlite3.

    This is deliberately defensive: avoids relying on SQLAlchemy, creates the
    products table with a minimal schema and writes using executemany if
    pandas.to_sql is not available for the connection.
    """
    try:
        import sqlite3 as _sqlite
        conn = _sqlite.connect(db_path)
        cur = conn.cursor()

        cur.execute('''
            CREATE TABLE IF NOT EXISTS products (
                ProdID INTEGER PRIMARY KEY,
                Name TEXT,
                Brand TEXT,
                Category TEXT,
                Rating REAL,
                ImageURL TEXT
            )
        ''')

        product_cols = ['ProdID', 'Name', 'Brand', 'Category', 'Rating', 'ImageURL']
        available_cols = [c for c in product_cols if c in train_data.columns]
        if not available_cols:
            print('✗ No product columns available for persistence')
            conn.close()
            return

        products_df = train_data[available_cols].drop_duplicates('ProdID')

        # Try pandas to_sql first with the connection (works with modern pandas)
        try:
            products_df.to_sql('products', conn, if_exists='replace', index=False)
            print(f'✓ Successfully persisted {len(products_df)} products to database')
            conn.close()
            return
        except Exception:
            # Fallback to manual insert
            cols_sql = ','.join(available_cols)
            placeholders = ','.join(['?'] * len(available_cols))
            rows = [tuple(r) for r in products_df.itertuples(index=False, name=None)]
            try:
                cur.execute('DELETE FROM products')
                cur.executemany(f'INSERT INTO products ({cols_sql}) VALUES ({placeholders})', rows)
                conn.commit()
                print(f'✓ Successfully persisted {len(rows)} products to database (fallback)')
            except Exception as e:
                print('Database persistence failed (fallback):', type(e).__name__, e)
            finally:
                conn.close()
    except Exception as e:
        print('Database persistence failed:', type(e).__name__, e)



def main(file_path: str | None = None):
    schema = None  # or provide a mapping as documented above
    path = _find_data_file(file_path)
    if path is None:
        print("No data file found. Pass --path /full/path/to/data.csv or place a data file named clean_data.csv or trending_products.csv next to this script.")
        sys.exit(2)

    print(f"Loading data from: {path}")
    # Show available data files in working directory to help debugging
    try:
        print("Available files in directory:")
        for f in os.listdir(Path(__file__).parent):
            if f.endswith(('.csv', '.tsv', '.xlsx')):
                print(f"  - {f}")
    except Exception:
        pass
    # assign to module-level `train_data` so other guarded blocks can access it
    global train_data
    train_data = load_and_clean_dataset(path, schema=schema, sep=None)

    # Basic cleaning already performed by loader; perform a quick validation summary
    train_data['Rating'] = train_data['Rating'].fillna(0)
    train_data['ReviewCount'] = train_data['ReviewCount'].fillna(0)
    train_data['Category'] = train_data['Category'].fillna('')
    train_data['Brand'] = train_data['Brand'].fillna('')
    train_data['Description'] = train_data['Description'].fillna('')

    required = ['ID','ProdID','Rating','ReviewCount','Category','Brand','Name','ImageURL','Description','Tags']
    summary = {}
    summary['shape'] = train_data.shape
    summary['dtypes'] = train_data[required].dtypes.astype(str).to_dict()
    summary['missing'] = train_data[required].isnull().sum().to_dict()
    summary['duplicates_rows'] = int(train_data.duplicated().sum())

    rating_min = float(train_data['Rating'].min()) if not train_data['Rating'].empty else 0.0
    rating_max = float(train_data['Rating'].max()) if not train_data['Rating'].empty else 0.0
    out_of_range = int(((train_data['Rating'] < 0) | (train_data['Rating'] > 5)).sum())
    summary['rating_min_max'] = (rating_min, rating_max)
    summary['rating_out_of_range'] = out_of_range

    summary['unique_users'] = int(train_data['ID'].nunique())
    summary['unique_items'] = int(train_data['ProdID'].nunique())

    print('Validation summary:')
    for k, v in summary.items():
        print(f'- {k}: {v}')

    # Additional validation and quick diagnostics
    validate_data_structure(train_data)

    # Run quick smoke tests for recommenders
    try:
        test_basic_recommendations(train_data)
    except Exception:
        print('Recommendation quick-tests failed')

    print('\nTop categories:')
    try:
        print(train_data['Category'].value_counts().head(5))
    except Exception:
        print('Could not compute top categories')

    print('\nTop brands:')
    try:
        print(train_data['Brand'].value_counts().head(5))
    except Exception:
        print('Could not compute top brands')

    num_users = train_data['ID'].nunique()
    num_items = train_data['ProdID'].nunique()
    num_ratings = train_data['Rating'].nunique()
    print(f"Number of unique users: {num_users}")
    print(f"Number of unique items: {num_items}")
    print(f"Number of unique ratings: {num_ratings}")
    # Run interactive/demo examples that require train_data
    try:
        run_examples(train_data)
    except Exception as e:
        print('Could not run examples:', e)
    # Pivot the DataFrame to create a heatmap of average rating per user (rows) across a small set of popular items (cols)
    # Select top-N items to keep the heatmap readable
    try:
        popular_items = train_data['ProdID'].value_counts().head(20).index
        subset = train_data[train_data['ProdID'].isin(popular_items)]
        heatmap_data = subset.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean')

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='coolwarm', cbar=True)
        plt.title('Average Ratings Heatmap (Top Items by Popularity)')
        plt.xlabel('ProdID')
        plt.ylabel('User ID')
        plt.tight_layout()
        plt.show()
    except Exception:
        print('Skipping heatmap (not enough data or plotting backend unavailable)')

    # Distribution of interactions
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        train_data['ID'].value_counts().hist(bins=10, edgecolor='k')
        plt.xlabel('Interactions per User')
        plt.ylabel('Number of Users')
        plt.title('Distribution of Interactions per User')

        plt.subplot(1, 2, 2)
        train_data['ProdID'].value_counts().hist(bins=10, edgecolor='k', color='green')
        plt.xlabel('Interactions per Item')
        plt.ylabel('Number of Items')
        plt.title('Distribution of Interactions per Item')

        plt.tight_layout()
        plt.show()
    except Exception:
        print('Skipping interaction distributions (plotting failed)')

    # Most popular items
    try:
        popular_items = train_data['ProdID'].value_counts().head(5)
        popular_items.plot(kind='bar', color='red')
        plt.title("Most Popular items")
        plt.show()
    except Exception:
        print('Skipping popular items plot')

    # Self-contained tag cleaning: no spaCy/sklearn required
    import re

    STOPWORDS = set(
        [
            'the','and','a','an','of','in','on','for','with','to','is','are','was','were','be','been','it','its','at','by','from','as','that','this','these','those',
            'or','but','not','no','so','if','then','than','too','very','can','could','should','would','will','just','about','over','under','into','out','up','down',
            'you','your','yours','me','my','mine','we','our','ours','they','their','theirs','he','him','his','she','her','hers','them','who','whom','which','what',
            'when','where','why','how','i','am','do','does','did','done','have','has','had','having','also','such'
        ]
    )


    def tokenize_clean(text: str):
        if not isinstance(text, str):
            text = '' if text is None else str(text)
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        return [t for t in tokens if t not in STOPWORDS]


    def clean_and_extract_tags(text):
        toks = tokenize_clean(text)
        return ", ".join(toks)

    columns_to_extract_tags_from = ['Category', 'Brand', 'Description']

    # Ensure columns exist and are strings
    for column in columns_to_extract_tags_from:
        if column not in train_data.columns:
            train_data[column] = ''
        else:
            train_data[column] = train_data[column].fillna('').astype(str)
        train_data[column] = train_data[column].apply(clean_and_extract_tags)


    # Concatenate the cleaned tags from all relevant columns
    train_data['Tags'] = train_data[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)

    average_ratings = train_data.groupby(['Name','ReviewCount','Brand','ImageURL'], dropna=False)['Rating'].mean().reset_index()

    top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)

    rating_base_recommendation = top_rated_items.head(10).copy()

    # Keep ratings and review counts as floats to preserve decimals
    rating_base_recommendation.loc[:, 'Rating'] = rating_base_recommendation['Rating'].round(2)
    try:
        rating_base_recommendation.loc[:, 'ReviewCount'] = rating_base_recommendation['ReviewCount'].astype(int)
    except Exception:
        # leave ReviewCount as-is if conversion fails
        pass

    print("Rating Base Recommendation System: (Trending Products)")
    print(rating_base_recommendation.loc[:, ['Name','Rating','ReviewCount','Brand','ImageURL']].to_string(index=False))
def run_examples(train_data: pd.DataFrame):
    """Run post-load examples and demos (content-based, collaborative, hybrid, evaluation).

    This function packages the large interactive/demo block so it only runs after
    `train_data` is available (i.e., inside main()).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
            cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

            item_name = 'OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'
            # find index safely
            if item_name in train_data['Name'].values:
                item_index = train_data[train_data['Name'] == item_name].index[0]
                similar_items = list(enumerate(cosine_similarities_content[item_index]))
                similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
                top_similar_items = similar_items[1:10]
                recommended_items_indics = [x[0] for x in top_similar_items]
                _ = train_data.iloc[recommended_items_indics][['Name', 'ReviewCount', 'Brand']]
            else:
                print(f"Demo item not found: {item_name}")
        except Exception as e:
            print('Skipping content-based demo (reason):', e)

        # content_based_recommendations moved to module scope for reuse

        # Example content-based usage
        try:
            item_name = 'OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'
            content_based_rec = content_based_recommendations(train_data, item_name, top_n=8)
            print(content_based_rec.head(5))
            # Demo: initialize DB engine (best-effort) and log 'view' interactions for displayed items
            try:
                engine = None
                try:
                    engine = init_db()
                except Exception:
                    engine = None

                if engine is not None and not content_based_rec.empty:
                    sample_user = globals().get('target_user_id', 0)
                    for _, row in content_based_rec.head(5).iterrows():
                        try:
                            prod_id = None
                            # common column names: 'Product' (collab) or lookup by Name
                            if 'Product' in row.index:
                                prod_id = int(row['Product'])
                            else:
                                name = row.get('Name')
                                if name is not None:
                                    candidate = train_data[train_data['Name'] == name]
                                    if not candidate.empty and 'ProdID' in candidate.columns:
                                        prod_id = int(candidate['ProdID'].iloc[0])
                            if prod_id is not None:
                                log_interaction(engine, sample_user, prod_id, 'view')
                        except Exception:
                            # best-effort logging; continue on errors
                            continue
            except Exception as _log_err:
                print('Interaction logging (content-based) skipped:', type(_log_err).__name__, _log_err)
        except Exception:
            print('Skipping content-based examples: train_data not available or error occurred')

        # Use floats (mean ratings) and avoid truncation
        try:
            user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0.0)

            # User-user cosine similarity on the float matrix
            user_similarity = cosine_similarity(user_item_matrix)
            np.fill_diagonal(user_similarity, 0.0)  # ignore self-similarity

            target_user_id = 4
            target_user_index = user_item_matrix.index.get_loc(target_user_id)

            user_similarities = user_similarity[target_user_index]

            # Weighted predicted scores for all items
            numerator = user_similarities @ user_item_matrix.values  # shape: (n_items,)
            denominator = np.abs(user_similarities).sum()
            if denominator == 0:
                predicted_scores = np.zeros_like(numerator)
            else:
                predicted_scores = numerator / denominator

            # Mask items already rated by the target user
            already_rated = user_item_matrix.iloc[target_user_index].values > 0
            predicted_scores_masked = np.where(already_rated, -np.inf, predicted_scores)

            # Top-N recommendations
            top_n = 10
            best_indices = np.argsort(predicted_scores_masked)[-top_n:][::-1]
            recommended_prod_ids = user_item_matrix.columns.values[best_indices]

            recommended_items_details = (train_data[train_data['ProdID'].isin(recommended_prod_ids)]
                                          .drop_duplicates(subset=['ProdID'])
                                          .set_index('ProdID')
                                          .loc[recommended_prod_ids, ['Name','ReviewCount','Brand','ImageURL','Rating']]
                                          .reset_index()
                                         )
            recommended_items_details.rename(columns={'ProdID':'Product'}, inplace=True)
            recommended_items_details['PredScore'] = np.round(predicted_scores[best_indices], 3)
            # Fill PredScore NaNs with item mean rating as a weak fallback
            try:
                if 'PredScore' in recommended_items_details.columns:
                    mask = recommended_items_details['PredScore'].isna()
                    if mask.any():
                        item_mean_map = train_data.groupby('ProdID')['Rating'].mean().to_dict()
                        # recommended_items_details has index 'Product' matching ProdID
                        recommended_items_details.loc[mask, 'PredScore'] = recommended_items_details.loc[mask, 'Product'].map(item_mean_map).fillna(0.0)
            except Exception:
                pass
            recommend_items = []
            for user_index in list(range(0, min(5, user_item_matrix.shape[0]))):
                rated_by_similar_user = user_item_matrix.iloc[user_index]
                not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
                recommend_items.extend(user_item_matrix.columns[not_rated_by_target_user][:10])

            recommended_items_details_demo = train_data[train_data['ProdID'].isin(recommend_items)][['Name','ReviewCount','Brand','ImageURL','Rating']]
            _ = recommended_items_details_demo.head(10)
            # Demo: log 'view' interactions for a few collaborative recommendations
            try:
                try:
                    engine = init_db()
                except Exception:
                    engine = None
                if engine is not None and not recommended_items_details_demo.empty:
                    sample_user = target_user_id if 'target_user_id' in globals() else 0
                    for _, r in recommended_items_details_demo.head(5).iterrows():
                        try:
                            # lookup ProdID by Name
                            name = r.get('Name')
                            if name is None:
                                continue
                            cand = train_data[train_data['Name'] == name]
                            if not cand.empty and 'ProdID' in cand.columns:
                                pid = int(cand['ProdID'].iloc[0])
                                log_interaction(engine, sample_user, pid, 'view')
                        except Exception:
                            continue
            except Exception as _log_err:
                print('Interaction logging (collaborative) skipped:', type(_log_err).__name__, _log_err)
        except Exception as e:
            print('Skipping collaborative demo (reason):', e)

        # Hybrid Recommendations (Combine Content-Based and Collaborative Filtering)
        def hybrid_recommendations(train_data,target_user_id, item_name, top_n=10):
            # Get content-based recommendations
            content_based_rec = content_based_recommendations(train_data,item_name, top_n)

            # Get collaborative filtering recommendations
            collaborative_filtering_rec = collaborative_filtering_recommendations(train_data,target_user_id, top_n)
            
            # Merge and deduplicate the recommendations
            hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec], ignore_index=True).drop_duplicates()
            # Ensure Product present
            if 'Product' not in hybrid_rec.columns and 'ProdID' in hybrid_rec.columns:
                hybrid_rec['Product'] = hybrid_rec['ProdID']
            # Fill missing Product via Name->ProdID map
            try:
                name_to_prod = dict(train_data[['Name','ProdID']].drop_duplicates().itertuples(index=False, name=None))
                if 'Product' in hybrid_rec.columns and 'Name' in hybrid_rec.columns:
                    hybrid_rec['Product'] = hybrid_rec['Product'].fillna(hybrid_rec['Name'].map(name_to_prod))
            except Exception:
                pass
            # Fill PredScore for CB rows (fallback to item mean)
            if 'PredScore' not in hybrid_rec.columns:
                hybrid_rec['PredScore'] = np.nan
            try:
                if 'Product' in hybrid_rec.columns:
                    item_mean_map = train_data.groupby('ProdID')['Rating'].mean().to_dict()
                    mask = hybrid_rec['PredScore'].isna()
                    hybrid_rec.loc[mask, 'PredScore'] = hybrid_rec.loc[mask, 'Product'].map(item_mean_map)
            except Exception:
                pass
            try:
                hybrid_rec['PredScore'] = hybrid_rec['PredScore'].astype(float).round(3)
            except Exception:
                pass
            return hybrid_rec.head(10)

        try:
            target_user_id = 4
            item_name = "OPI Nail Lacquer Polish .5oz/15mL - This Gown Needs A Crown NL U11"
            hybrid_rec = hybrid_recommendations(train_data, target_user_id, item_name, top_n=10)
            print(hybrid_rec.head(5))
        except Exception:
            print('Skipping hybrid demo')

        # === Evaluation Utilities ===
        from collections import defaultdict
        from typing import List, Tuple, Dict, Set

        def train_test_split_by_user(df: pd.DataFrame, test_size: int = 1, min_items: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
            # If you have timestamps, replace the sort with sort_values(['ID', 'timestamp'])
            df_sorted = df.sort_values(['ID', 'ProdID']).reset_index(drop=True)
            train_rows, test_rows = [], []
            for uid, group in df_sorted.groupby('ID'):
                if len(group) < min_items:
                    train_rows.append(group)
                    continue
                test = group.tail(test_size)
                train = group.iloc[:-test_size]
                train_rows.append(train)
                test_rows.append(test)
            train_df = pd.concat(train_rows).reset_index(drop=True)
            test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=df.columns)
            return train_df, test_df

        def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
            return df.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0.0)

        def cf_predict_scores(user_item: pd.DataFrame):
            sim = cosine_similarity(user_item)
            np.fill_diagonal(sim, 0.0)
            numerators = sim @ user_item.values
            denom = np.abs(sim).sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            preds = numerators / denom
            return preds, sim

        def recommend_top_k(user_item: pd.DataFrame, preds: np.ndarray, k: int = 10) -> Dict[int, List[int]]:
            already = user_item.values > 0
            masked = np.where(already, -np.inf, preds)
            top_idx = np.argsort(masked, axis=1)[:, -k:][:, ::-1]
            prod_ids = user_item.columns.values
            user_ids = user_item.index.values
            return {int(user_ids[i]): [int(pid) for pid in prod_ids[top_idx[i]]] for i in range(len(user_ids))}

        def evaluate_at_k(recs: Dict[int, List[int]], test_df: pd.DataFrame, k: int, catalog: Set[int]) -> Dict[str, float]:
            truth: Dict[int, Set[int]] = defaultdict(set)
            for row in test_df[['ID','ProdID']].itertuples(index=False):
                truth[int(row.ID)].add(int(row.ProdID))

            precisions, recalls = [], []
            hits = 0
            recommended_items_global: Set[int] = set()

            for uid, rec_list in recs.items():
                gt = truth.get(uid, set())
                if not gt:
                    continue
                rec_at_k = rec_list[:k]
                recommended_items_global.update(rec_at_k)
                tp = len(set(rec_at_k) & gt)
                precisions.append(tp / max(k, 1))
                recalls.append(tp / max(len(gt), 1))
                hits += 1 if tp > 0 else 0

            num_eval_users = max(len([u for u in truth if truth[u]]), 1)
            precision_k = float(np.mean(precisions)) if precisions else 0.0
            recall_k = float(np.mean(recalls)) if recalls else 0.0
            hit_rate = hits / num_eval_users
            coverage = len(recommended_items_global) / max(len(catalog), 1)
            return {'precision@k': round(precision_k, 4), 'recall@k': round(recall_k, 4), 'hit_rate': round(hit_rate, 4), 'item_coverage': round(coverage, 4)}

        # === Run Evaluation ===
        try:
            train_df, test_df = train_test_split_by_user(train_data, test_size=1, min_items=2)
            user_item = build_user_item_matrix(train_df)
            preds, _ = cf_predict_scores(user_item)
            all_recs = recommend_top_k(user_item, preds, k=50)
            catalog = set(train_df['ProdID'].astype(int).unique())
            for k_val in (5, 10):
                metrics = evaluate_at_k(all_recs, test_df, k=k_val, catalog=catalog)
                print(f"K={k_val} -> precision@k={metrics['precision@k']}, recall@k={metrics['recall@k']}, hit_rate={metrics['hit_rate']}, item_coverage={metrics['item_coverage']}")

            sample_uid = next(iter(all_recs.keys()), None)
            if sample_uid is not None:
                sample_prod_ids = all_recs[sample_uid][:10]
                # Map product IDs to names when possible for readability
                prod_to_name = dict(train_df[['ProdID','Name']].drop_duplicates().itertuples(index=False, name=None))
                named = [prod_to_name.get(int(pid), pid) for pid in sample_prod_ids]
                print(f"\nSample recommendations for user {sample_uid} (names if available): {named}")
        except Exception as e:
            print("Evaluation could not run - missing prerequisites or error:", type(e).__name__, e)
    except Exception as _demo_error:
        print('Skipping large demo/eval block (reason):', _demo_error)

    # end run_examples

    
    

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    # Build float user-item matrix of mean ratings
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0.0)
    if target_user_id not in user_item_matrix.index:
        print(f"User {target_user_id} not found.")
        return pd.DataFrame()

    # User-user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    np.fill_diagonal(user_similarity, 0.0)

    # Locate target user
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]

    # Weighted prediction for each item
    numerator = user_similarities @ user_item_matrix.values
    denominator = np.abs(user_similarities).sum()
    predicted_scores = numerator / denominator if denominator != 0 else np.zeros_like(numerator)

    # If CF produced all zeros (cold-start or no similar users), fall back to item mean rating
    try:
        if np.allclose(predicted_scores, 0.0):
            item_means = train_data.groupby('ProdID')['Rating'].mean().reindex(user_item_matrix.columns).fillna(0).values
            # use normalized item means as a weak signal
            predicted_scores = item_means
    except Exception:
        pass

    # Mask already-rated items
    already_rated = user_item_matrix.iloc[target_user_index].values > 0
    predicted_scores_masked = np.where(already_rated, -np.inf, predicted_scores)

    # Top-N items by predicted score
    best_indices = np.argsort(predicted_scores_masked)[-top_n:][::-1]
    recommended_prod_ids = user_item_matrix.columns.values[best_indices]

    # Join back to product metadata
    rec_df = (train_data[train_data['ProdID'].isin(recommended_prod_ids)]
              .drop_duplicates(subset=['ProdID'])
              .set_index('ProdID')
              .loc[recommended_prod_ids, ['Name','ReviewCount','Brand','ImageURL','Rating']]
              .reset_index())
    rec_df.rename(columns={'ProdID':'Product'}, inplace=True)
    rec_df['PredScore'] = np.round(predicted_scores[best_indices], 3)
    return rec_df.head(top_n)


def content_based_recommendations(train_data: pd.DataFrame, item_name: str, top_n: int = 10) -> pd.DataFrame:
    """Return top-N items most similar to item_name using TF-IDF over the `Tags` column.

    This function is safe to call when `train_data` is available. It returns an empty
    DataFrame if the item is not found.
    """
    if not isinstance(train_data, pd.DataFrame) or train_data.empty:
        return pd.DataFrame()

    if item_name not in train_data['Name'].values:
        # item not found
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Tags'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    item_index = int(train_data[train_data['Name'] == item_name].index[0])
    scores = list(enumerate(cosine_sim[item_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores[1: top_n + 1]
    indices = [i for i, _ in top_scores]

    # Build a product-level view: deduplicate by ProdID and aggregate Rating/ReviewCount
    recs = train_data.iloc[indices].copy()
    if 'ProdID' in recs.columns:
        recs['Rating'] = recs.groupby('ProdID')['Rating'].transform('mean')
        recs['ReviewCount'] = recs.groupby('ProdID')['ReviewCount'].transform('mean')
        recs = recs.drop_duplicates(subset=['ProdID'])
    # return consistent columns; include ProdID standardized as Product
    out_cols = [c for c in ['ProdID', 'Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'] if c in recs.columns]
    out = recs[out_cols].reset_index(drop=True).head(top_n)
    if 'ProdID' in out.columns:
        out = out.rename(columns={'ProdID': 'Product'})
    return out

# Hybrid Recommendations (Combine Content-Based and Collaborative Filtering)
def hybrid_recommendations(train_data,target_user_id, item_name, top_n=10):
    # Get content-based recommendations
    content_based_rec = content_based_recommendations(train_data,item_name, top_n)

    # Get collaborative filtering recommendations
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data,target_user_id, top_n)
    
    # Merge and deduplicate the recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec], ignore_index=True).drop_duplicates()
    # Ensure Product column present
    if 'Product' not in hybrid_rec.columns and 'ProdID' in hybrid_rec.columns:
        hybrid_rec['Product'] = hybrid_rec['ProdID']
    # Fill missing Product via Name->ProdID mapping
    try:
        name_to_prod = dict(train_data[['Name','ProdID']].drop_duplicates().itertuples(index=False, name=None))
        if 'Product' in hybrid_rec.columns and 'Name' in hybrid_rec.columns:
            hybrid_rec['Product'] = hybrid_rec['Product'].fillna(hybrid_rec['Name'].map(name_to_prod))
    except Exception:
        pass
    # PredScore fallback for CB rows using item mean rating
    if 'PredScore' not in hybrid_rec.columns:
        hybrid_rec['PredScore'] = np.nan
    try:
        if 'Product' in hybrid_rec.columns:
            item_mean_map = train_data.groupby('ProdID')['Rating'].mean().to_dict()
            mask = hybrid_rec['PredScore'].isna()
            hybrid_rec.loc[mask, 'PredScore'] = hybrid_rec.loc[mask, 'Product'].map(item_mean_map)
    except Exception:
        pass
    try:
        hybrid_rec['PredScore'] = hybrid_rec['PredScore'].astype(float).round(3)
    except Exception:
        pass
    return hybrid_rec.head(10)

# === Evaluation Utilities ===
# Train/test split by user (leave-N-out deterministic)
from collections import defaultdict
from typing import List, Tuple, Dict, Set


def train_test_split_by_user(df: pd.DataFrame, test_size: int = 1, min_items: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # If you have timestamps, replace the sort with sort_values(['ID', 'timestamp'])
    df_sorted = df.sort_values(['ID', 'ProdID']).reset_index(drop=True)
    train_rows, test_rows = [], []
    for uid, group in df_sorted.groupby('ID'):
        if len(group) < min_items:
            train_rows.append(group)
            continue
        test = group.tail(test_size)
        train = group.iloc[:-test_size]
        train_rows.append(train)
        test_rows.append(test)
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0.0)


def cf_predict_scores(user_item: pd.DataFrame):
    sim = cosine_similarity(user_item)
    np.fill_diagonal(sim, 0.0)
    numerators = sim @ user_item.values
    denom = np.abs(sim).sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    preds = numerators / denom
    return preds, sim


def recommend_top_k(user_item: pd.DataFrame, preds: np.ndarray, k: int = 10) -> Dict[int, List[int]]:
    already = user_item.values > 0
    masked = np.where(already, -np.inf, preds)
    top_idx = np.argsort(masked, axis=1)[:, -k:][:, ::-1]
    prod_ids = user_item.columns.values
    user_ids = user_item.index.values
    return {int(user_ids[i]): [int(pid) for pid in prod_ids[top_idx[i]]] for i in range(len(user_ids))}


def evaluate_at_k(recs: Dict[int, List[int]], test_df: pd.DataFrame, k: int, catalog: Set[int]) -> Dict[str, float]:
    truth: Dict[int, Set[int]] = defaultdict(set)
    for row in test_df[['ID','ProdID']].itertuples(index=False):
        truth[int(row.ID)].add(int(row.ProdID))

    precisions, recalls = [], []
    hits = 0
    recommended_items_global: Set[int] = set()

    for uid, rec_list in recs.items():
        gt = truth.get(uid, set())
        if not gt:
            continue
        rec_at_k = rec_list[:k]
        recommended_items_global.update(rec_at_k)
        tp = len(set(rec_at_k) & gt)
        precisions.append(tp / max(k, 1))
        recalls.append(tp / max(len(gt), 1))
        hits += 1 if tp > 0 else 0

    num_eval_users = max(len([u for u in truth if truth[u]]), 1)
    precision_k = float(np.mean(precisions)) if precisions else 0.0
    recall_k = float(np.mean(recalls)) if recalls else 0.0
    hit_rate = hits / num_eval_users
    coverage = len(recommended_items_global) / max(len(catalog), 1)
    return {'precision@k': round(precision_k, 4), 'recall@k': round(recall_k, 4), 'hit_rate': round(hit_rate, 4), 'item_coverage': round(coverage, 4)}

# === Run Evaluation ===
# Note: Evaluation is performed inside run_examples() after data is loaded.

    # === Visualize Recommendations (polished) ===
from IPython.display import Image, display, HTML
from urllib.parse import urlparse
import math


def _is_valid_url(u: str) -> bool:
    try:
        parsed = urlparse(str(u))
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def show_recommendations(df, title: str = "Recommended for you", image_col: str = 'ImageURL', name_col: str = 'Name', score_col: str | None = None, max_items: int = 12, per_row: int = 6, img_width: int = 140):
    if df is None or len(df) == 0:
        display(HTML('<p style="color:#999">No recommendations to display.</p>'))
        return
    # Show name, image, and score column (if present). Also include Rating if available so we can display N/A for zeros.
    cols = [c for c in [name_col, image_col, score_col, 'Rating'] if c and c in df.columns]
    data = df[cols].head(max_items).copy()

    # Build simple grid with inline images and captions
    items_html = []
    for _, row in data.iterrows():
        name = str(row.get(name_col, ""))
        url = row.get(image_col, "")
        score = row.get(score_col) if score_col and score_col in row else None
        score_html = ''
        if isinstance(score, (int, float)):
            score_html = f'<div class="score">Score: {score:.3f}</div>'
        img_html = f'<img src="{url}" width="{img_width}" />' if _is_valid_url(url) else '<div class="noimg">No image</div>'
        # Rating display: treat 0.0 as N/A for presentation
        rating_display = ''
        if 'Rating' in row.index:
            try:
                r = row.get('Rating')
                if pd.isna(r) or float(r) == 0.0:
                    rating_display = '<div class="score">Rating: N/A</div>'
                else:
                    rating_display = f'<div class="score">Rating: {float(r):.2f}</div>'
            except Exception:
                rating_display = ''
        # Prefer showing PredScore or score_col; otherwise show rating_display.
        if not score_html:
            score_html = rating_display
        else:
            score_html = score_html + rating_display
        items_html.append(f'<div class="card">{img_html}<div class="name">{name}</div>{score_html}</div>')

    rows = []
    for i in range(0, len(items_html), per_row):
        row_html = ''.join(items_html[i:i+per_row])
        rows.append(f'<div class="row">{row_html}</div>')

    html = f'''
    <style>
    .rec-container {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
    .rec-title {{ font-size: 18px; font-weight: 600; margin: 8px 0 12px; }}
    .row {{ display: flex; flex-wrap: nowrap; gap: 12px; margin-bottom: 12px; }}
    .card {{ width: {img_width+10}px; text-align: center; font-size: 12px; color: #333; }}
    .card img {{ border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.15); }}
    .card .name {{ margin-top: 6px; height: 32px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }}
    .card .score {{ margin-top: 2px; color: #666; font-variant-numeric: tabular-nums; }}
    .noimg {{ width: {img_width}px; height: {int(img_width*1.0)}px; display:flex; align-items:center; justify-content:center; background:#f3f4f6; color:#999; border-radius:6px; }}
    </style>
    <div class="rec-container">
      <div class="rec-title">{title}</div>
      {''.join(rows)}
    </div>
    '''
    display(HTML(html))


# Try to auto-detect a recommendations DataFrame
    # Note: Visualization and LLM explanation pipelines are invoked from run_examples() or __main__ when requested.

    # --- Unified Gemini init with fallbacks (google-genai preferred, fallback to google-generativeai) ---
import os
from typing import Optional, Tuple

def _ensure_api_key_env():
    # Prefer GEMINI_API_KEY; if missing but GOOGLE_API_KEY present, mirror it
    if not os.getenv('GEMINI_API_KEY') and os.getenv('GOOGLE_API_KEY'):
        os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_API_KEY']
    has_key = bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))
    print('GEMINI_API_KEY set:', bool(os.getenv('GEMINI_API_KEY')), '| GOOGLE_API_KEY set:', bool(os.getenv('GOOGLE_API_KEY')))
    return has_key

class _GenAIClientWrapper:
    def __init__(self, client=None, model_name: str = '', sdk: str = '', disable_thinking: bool = False):
        self.client = client
        self.model_name = model_name
        self.sdk = sdk  # 'google-genai' or 'google-generativeai'
        self.disable_thinking = disable_thinking

def init_gemini(prefer_pro: bool = False, disable_thinking: bool = False) -> Tuple[Optional[_GenAIClientWrapper], str, Optional[Exception]]:
    """
    Initialize Gemini with model fallbacks.
    - Tries google-genai first (2.5 series), then falls back to google-generativeai (1.5 series).
    - prefer_pro: prioritize Pro model when available.
    - disable_thinking: for 2.5 models, set thinking_budget=0 to reduce latency/cost.
    Returns (client_wrapper, used_model, error).
    """
    if not _ensure_api_key_env():
        return None, '', RuntimeError('No GEMINI_API_KEY/GOOGLE_API_KEY found')
    # Candidate lists
    flash_25 = ['gemini-2.5-flash']
    pro_25 = ['gemini-2.5-pro']
    flash_15 = ['gemini-1.5-flash', 'gemini-1.5-flash-latest']
    pro_15 = ['gemini-1.5-pro', 'gemini-1.5-pro-latest', 'gemini-pro']
    ordered_models = []
    if prefer_pro:
        ordered_models.extend(pro_25 + flash_25 + pro_15 + flash_15)
    else:
        ordered_models.extend(flash_25 + pro_25 + flash_15 + pro_15)

    last_error: Optional[Exception] = None

    # Try new google-genai SDK (2.5)
    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore
        client = genai.Client()  # picks up GEMINI_API_KEY
        # Pick first 2.5 model present in ordered list
        for m in ordered_models:
            if not m.startswith('gemini-2.5'):
                continue
            # Store wrapper and return
            wrapper = _GenAIClientWrapper(client=client, model_name=m, sdk='google-genai', disable_thinking=disable_thinking)
            return wrapper, m, None
    except Exception as e:
        last_error = e

    # Fallback to legacy google-generativeai (1.5)
    try:
        import google.generativeai as genai_old  # type: ignore
        genai_old.configure(api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))
        for m in ordered_models:
            if not m.startswith('gemini-1.5') and m != 'gemini-pro':
                continue
            model = genai_old.GenerativeModel(m)
            wrapper = _GenAIClientWrapper(client=model, model_name=m, sdk='google-generativeai', disable_thinking=False)
            return wrapper, m, None
    except Exception as e2:
        last_error = e2

    return None, '', last_error

def explain_recommendation_unified(client_wrapper: _GenAIClientWrapper, user_id: int, product_name: str, reason_features: str = '') -> str:
    """Generate a short explanation using the available SDK without raising exceptions."""
    if client_wrapper is None:
        return 'Explanation unavailable (no client)'
    # Simple sqlite cache: table llm_explanations(prompt_hash TEXT PRIMARY KEY, explanation TEXT)
    try:
        import sqlite3 as _sqlite
        conn = _sqlite.connect('recommender.db')
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS llm_explanations (prompt_hash TEXT PRIMARY KEY, explanation TEXT)')
        conn.commit()
    except Exception:
        conn = None
    prompt = f"User {user_id} may like {product_name}. {reason_features}\nExplain in one concise sentence for a shopper."
    try:
        if client_wrapper.sdk == 'google-genai':
            from google.genai import types as genai_types  # type: ignore
            cfg = None
            if client_wrapper.disable_thinking and '2.5' in client_wrapper.model_name:
                try:
                    cfg = genai_types.GenerateContentConfig(thinking_config=genai_types.ThinkingConfig(thinking_budget=0))
                except Exception:
                    cfg = None
            if cfg is not None:
                resp = client_wrapper.client.models.generate_content(model=client_wrapper.model_name, contents=prompt, config=cfg)
            else:
                resp = client_wrapper.client.models.generate_content(model=client_wrapper.model_name, contents=prompt)
            out = getattr(resp, 'text', None) or (getattr(resp, 'candidates', [None])[0].content.parts[0].text if getattr(resp, 'candidates', None) else 'Explanation unavailable')
            # write to cache
            try:
                if conn is not None:
                    import hashlib
                    key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                    cur.execute('INSERT OR REPLACE INTO llm_explanations (prompt_hash, explanation) VALUES (?,?)', (key, out))
                    conn.commit()
            except Exception:
                pass
            return out
        else:  # google-generativeai
            resp = client_wrapper.client.generate_content(prompt)
            out = getattr(resp, 'text', None) or 'Explanation unavailable'
            try:
                if conn is not None:
                    import hashlib
                    key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                    cur.execute('INSERT OR REPLACE INTO llm_explanations (prompt_hash, explanation) VALUES (?,?)', (key, out))
                    conn.commit()
            except Exception:
                pass
            return out
    except Exception as ex:
        return f'Explanation unavailable ({type(ex).__name__})'
    
    # Note: LLM initialization is performed on-demand in __main__ when --enable-llm is provided.

class RecommendationExplainer:
    """Multi-provider LLM explainer with sqlite caching and safe fallbacks.

    Providers attempted in order: Gemini (google-genai), OpenAI, Ollama(local).
    If no provider is available, falls back to rule-based templates.
    """
    def __init__(self, db_path: str = 'recommender.db'):
        import sqlite3 as _sqlite
        self.db_path = db_path
        self.client = None
        self.provider = None
        # init cache
        try:
            conn = _sqlite.connect(self.db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS llm_explanations (
                    prompt_hash TEXT PRIMARY KEY,
                    explanation TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception:
            pass
        # init providers
        self._init_provider()

    def _init_provider(self) -> None:
        if self._try_gemini():
            return
        if self._try_openai():
            return
        if self._try_ollama():
            return
        print('No LLM provider available; using rule-based explanations')

    def _try_gemini(self) -> bool:
        try:
            from google import genai  # type: ignore
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                return False
            # Client picks up GEMINI_API_KEY env var or use explicit
            try:
                os.environ.setdefault('GEMINI_API_KEY', api_key)
            except Exception:
                pass
            self.client = genai.Client()
            self.provider = 'gemini'
            print('LLM: Gemini initialized')
            return True
        except Exception:
            return False

    def _try_openai(self) -> bool:
        try:
            import openai  # type: ignore
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return False
            openai.api_key = api_key
            self.client = openai
            self.provider = 'openai'
            print('LLM: OpenAI initialized')
            return True
        except Exception:
            return False

    def _try_ollama(self) -> bool:
        try:
            import requests  # type: ignore
            r = requests.get('http://localhost:11434/api/tags', timeout=2)
            if getattr(r, 'status_code', 0) == 200:
                self.client = 'ollama'
                self.provider = 'ollama'
                print('LLM: Ollama (local) available')
                return True
            return False
        except Exception:
            return False

    def _get_cached(self, prompt: str) -> Optional[str]:
        try:
            import sqlite3 as _sqlite, hashlib as _hash
            key = _hash.sha256(prompt.encode('utf-8')).hexdigest()
            conn = _sqlite.connect(self.db_path)
            cur = conn.cursor()
            cur.execute('SELECT explanation FROM llm_explanations WHERE prompt_hash=?', (key,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None

    def _cache(self, prompt: str, explanation: str) -> None:
        try:
            import sqlite3 as _sqlite, hashlib as _hash
            key = _hash.sha256(prompt.encode('utf-8')).hexdigest()
            conn = _sqlite.connect(self.db_path)
            conn.execute('INSERT OR REPLACE INTO llm_explanations (prompt_hash, explanation) VALUES (?,?)', (key, explanation))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def explain(self, user_id: int, product_name: str, features: str = '', user_history: str = '') -> str:
        prompt = (
            'You are an e-commerce recommendation assistant. Explain why this product is recommended.\n\n'
            f'User ID: {user_id}\n'
            f'Product: {product_name}\n'
            f'Matching Features: {features}\n'
            f'User History: {user_history or "Limited history available"}\n\n'
            'Provide a natural, one-sentence explanation for a shopper. Be specific and helpful.'
        )
        cached = self._get_cached(prompt)
        if cached:
            return cached
        explanation = self._generate(prompt, product_name, features)
        self._cache(prompt, explanation)
        return explanation

    def _generate(self, prompt: str, product_name: str, features: str) -> str:
        # Gemini (google-genai)
        if self.provider == 'gemini':
            try:
                resp = self.client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                text = getattr(resp, 'text', None)
                return (text or '').strip() or self._rule_based(product_name, features)
            except Exception:
                pass
        # OpenAI (chat completions)
        if self.provider == 'openai':
            try:
                resp = self.client.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=100,
                )
                return (resp.choices[0].message.content or '').strip()
            except Exception:
                pass
        # Ollama local
        if self.provider == 'ollama':
            try:
                import requests  # type: ignore
                r = requests.post('http://localhost:11434/api/generate', json={'model': 'llama2', 'prompt': prompt, 'stream': False}, timeout=8)
                data = r.json() if hasattr(r, 'json') else {}
                return str(data.get('response', '')).strip() or self._rule_based(product_name, features)
            except Exception:
                pass
        return self._rule_based(product_name, features)

    @staticmethod
    def _rule_based(product_name: str, features: str) -> str:
        templates = [
            f"This product matches your preferences based on {features} and has positive ratings from similar shoppers.",
            f"Recommended because it shares key attributes with products you've viewed: {features}.",
            f"Based on your shopping patterns, this {product_name} aligns well with your interests in {features}.",
        ]
        try:
            idx = abs(hash(product_name)) % len(templates)
        except Exception:
            idx = 0
        return templates[idx]

    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E-commerce recommendation data loader + overview')
    parser.add_argument('--path', '-p', help='Path to CSV/TSV data file', default=None)
    parser.add_argument('--enable-llm', action='store_true', help='Enable LLM explanations (requires configured API key)')
    parser.add_argument('--db-path', help='SQLite DB path to use for persistence', default='recommender.db')
    args = parser.parse_args()

    # Initialize DB if SQLAlchemy is present; otherwise proceed without DB persistence
    engine = None
    try:
        engine = init_db(args.db_path)
        print(f"DB initialized at: {args.db_path}")
    except Exception as _e:
        print('DB init failed, continuing without persistence:', type(_e).__name__, _e)

    # Run main load & demos
    main(args.path)

    # After main() completes, persist products (best-effort)
    try:
        if 'train_data' in globals() and isinstance(train_data, pd.DataFrame) and not train_data.empty:
            products_df = train_data[['ProdID','Name','ImageURL','Brand','Category','Rating']].drop_duplicates(subset=['ProdID'])
            if engine is not None:
                try:
                    persist_products_sqlite(products_df, engine)
                    print('Product metadata persisted to DB (via SQLAlchemy)')
                except Exception as _e:
                    print('persist_products_sqlite failed, falling back to safe_persist_products:', type(_e).__name__, _e)
                    safe_persist_products(products_df, db_path=args.db_path)
            else:
                # No SQLAlchemy engine available; use sqlite3-backed fallback
                safe_persist_products(products_df, db_path=args.db_path)
    except Exception as _e:
        print('Could not persist products:', type(_e).__name__, _e)

    # If LLM flag is set, show a small explanation sample for the first recommendation
    if args.enable_llm:
        try:
            explainer = RecommendationExplainer(db_path=args.db_path)
            if 'train_data' in globals() and isinstance(train_data, pd.DataFrame) and not train_data.empty:
                sample_row = train_data.iloc[0]
                features = ', '.join([str(sample_row.get('Brand','')), str(sample_row.get('Category',''))]).strip(', ')
                expl = explainer.explain(int(sample_row['ID']), str(sample_row['Name']), features=features)
                print('\nLLM explanation sample (multi-provider):')
                print(expl)
        except Exception as _e:
            print('LLM explanation failed (skipping):', type(_e).__name__, _e)


