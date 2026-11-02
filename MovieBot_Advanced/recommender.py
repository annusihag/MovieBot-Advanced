import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _parse_json_list(value):
    """Parse Kaggle TMDB JSON-string columns safely."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        try:
            result = ast.literal_eval(value)
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []


def _names(value, limit=None):
    items = _parse_json_list(value)
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            result.append(str(item["name"]))
    if limit:
        result = result[:limit]
    return result


def _extract_director(crew_value):
    for person in _parse_json_list(crew_value):
        if isinstance(person, dict) and person.get("job") == "Director":
            return str(person.get("name", ""))
    return ""


def _safe_number(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def load_movies(data_dir):
    """
    Load the official Kaggle TMDB 5000 files.
    Expected filenames:
      data/tmdb_5000_movies.csv
      data/tmdb_5000_credits.csv
    """
    data_dir = Path(data_dir)
    movies_path = data_dir / "tmdb_5000_movies.csv"
    credits_path = data_dir / "tmdb_5000_credits.csv"

    if not movies_path.exists() or not credits_path.exists():
        raise FileNotFoundError(
            "Missing Kaggle data files. Place tmdb_5000_movies.csv and "
            "tmdb_5000_credits.csv inside the project's data/ folder."
        )

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    if "id" not in movies.columns or "movie_id" not in credits.columns:
        raise ValueError("Unexpected dataset schema: expected movies.id and credits.movie_id.")

    credits = credits.copy()
    credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    df = movies.merge(
        credits[["movie_id", "cast", "crew"]],
        left_on="id", right_on="movie_id", how="left"
    )

    # Clean JSON columns into readable fields.
    df["genres_list"] = df.get("genres", pd.Series("", index=df.index)).apply(_names)
    df["genres_text"] = df["genres_list"].apply(lambda x: " ".join(x))
    df["keywords_list"] = df.get("keywords", pd.Series("", index=df.index)).apply(_names)
    df["keywords_text"] = df["keywords_list"].apply(lambda x: " ".join(x))
    df["cast_list"] = df.get("cast", pd.Series("", index=df.index)).apply(lambda x: _names(x, 5))
    df["cast_text"] = df["cast_list"].apply(lambda x: ", ".join(x))
    df["director"] = df.get("crew", pd.Series("", index=df.index)).apply(_extract_director)

    if "title" not in df.columns:
        df["title"] = df.get("original_title", "Unknown title")
    df["title"] = df["title"].fillna(df.get("original_title", "Unknown title")).astype(str)
    for col in ["overview", "tagline", "original_title"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    else:
        df["release_year"] = np.nan

    if "vote_average" not in df.columns:
        df["vote_average"] = 0
    if "vote_count" not in df.columns:
        df["vote_count"] = 0
    if "popularity" not in df.columns:
        df["popularity"] = 0

    df["vote_average"] = _safe_number(df["vote_average"])
    df["vote_count"] = _safe_number(df["vote_count"])
    df["popularity"] = _safe_number(df["popularity"])
    df["movie_id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["movie_id"]).drop_duplicates("movie_id").reset_index(drop=True)

    # Weighted text corpus: synopsis, genres, keywords, cast, director and tagline.
    df["content_text"] = (
        df["title"].fillna("") + " " +
        df["genres_text"].fillna("") + " " +
        df["genres_text"].fillna("") + " " +
        df["keywords_text"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["cast_text"].fillna("") + " " +
        df["director"].fillna("") + " " +
        df["tagline"].fillna("")
    ).str.lower()

    # Optional poster URL support: the public Kaggle CSV usually does not contain poster paths.
    if "poster_url" not in df.columns:
        df["poster_url"] = ""

    return df


class MovieRecommender:
    def __init__(self, df):
        if df.empty:
            raise ValueError("The movie dataset is empty.")
        self.df = df.reset_index(drop=True)
        self.df["movie_id"] = pd.to_numeric(self.df["movie_id"], errors="coerce")
        self.id_to_index = {
            int(movie_id): i
            for i, movie_id in enumerate(self.df["movie_id"])
            if pd.notna(movie_id)
        }
        self.title_lookup = {}
        for i, title in enumerate(self.df["title"].astype(str)):
            key = self._normalize(title)
            self.title_lookup.setdefault(key, []).append(i)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
        )
        self.matrix = self.vectorizer.fit_transform(self.df["content_text"].fillna(""))
        # Sparse TF-IDF matrix is retained; pairwise scores are calculated on demand.
        self.feature_names = set(self.vectorizer.get_feature_names_out())

    @staticmethod
    def _normalize(text):
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def all_genres(self):
        return sorted({
            genre
            for genres in self.df["genres_list"]
            for genre in genres
            if genre
        })

    def frame_by_ids(self, ids):
        valid = []
        for value in ids or []:
            try:
                idx = self.id_to_index.get(int(value))
                if idx is not None:
                    valid.append(idx)
            except (TypeError, ValueError):
                continue
        if not valid:
            return self.df.iloc[0:0]
        return self.df.iloc[valid].copy()

    def _rank_indices(self, scores, excluded=None, top_n=6):
        excluded = set(excluded or [])
        order = np.argsort(scores)[::-1]
        picked = []
        for idx in order:
            if idx in excluded:
                continue
            if scores[idx] <= 0:
                continue
            picked.append(int(idx))
            if len(picked) >= top_n:
                break
        return picked

    def recommend_similar(self, title, top_n=6):
        query = self._normalize(title)
        if not query:
            return self.df.iloc[0:0].copy(), "Please enter a movie title."

        exact = self.title_lookup.get(query, [])
        if not exact:
            # Partial title lookup, preferring the shortest matching title.
            partial = [
                (len(self._normalize(self.df.iloc[i]["title"])), i)
                for i, t in enumerate(self.df["title"].astype(str))
                if query in self._normalize(t)
            ]
            if partial:
                exact = [sorted(partial)[0][1]]

        if not exact:
            # If it is not a title, treat it as a natural-language request.
            return self.recommend_from_text(title, top_n)

        idx = exact[0]
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        indices = self._rank_indices(scores, excluded={idx}, top_n=top_n)
        results = self.df.iloc[indices].copy()
        return results, f"Because you mentioned **{self.df.iloc[idx]['title']}**, here are movies with similar metadata and themes."

    def recommend_from_text(self, query, top_n=6):
        raw = str(query).strip()
        if not raw:
            return self.df.iloc[0:0].copy(), "Tell me a genre, mood, actor, director, or movie title."

        normalized = self._normalize(raw)
        # If the user names a movie inside a longer sentence, use title similarity.
        for title in sorted(self.title_lookup, key=len, reverse=True):
            if title and re.search(r"\b" + re.escape(title) + r"\b", normalized):
                idx = self.title_lookup[title][0]
                scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
                indices = self._rank_indices(scores, excluded={idx}, top_n=top_n)
                results = self.df.iloc[indices].copy()
                return results, f"I found **{self.df.iloc[idx]['title']}** in your request. Try these similar films."

        query_vector = self.vectorizer.transform([normalized])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = self._rank_indices(scores, top_n=top_n)

        # Add a simple popularity prior only to break close ties.
        if not indices:
            # Keyword fallback for genre words and common moods.
            aliases = {
                "funny": "comedy", "hilarious": "comedy", "scary": "horror",
                "frightening": "horror", "love": "romance", "romantic": "romance",
                "space": "science fiction", "sci fi": "science fiction",
                "mind bending": "thriller", "sad": "drama", "emotional": "drama",
                "fight": "action", "fighting": "action", "mystery": "mystery",
            }
            for phrase, replacement in aliases.items():
                if phrase in normalized:
                    normalized += " " + replacement
            query_vector = self.vectorizer.transform([normalized])
            scores = cosine_similarity(query_vector, self.matrix).ravel()
            indices = self._rank_indices(scores, top_n=top_n)

        if not indices:
            return self.df.iloc[0:0].copy(), (
                "I couldn't find a strong match. Try a known movie title or a genre "
                "such as action, comedy, drama, romance, thriller, or science fiction."
            )

        results = self.df.iloc[indices].copy()
        return results, f"Here are **{len(results)} recommendations** matched to your request using movie metadata and text similarity."

    def search_catalog(self, query, catalog=None, top_n=60):
        catalog = self.df if catalog is None else catalog
        q = self._normalize(query)
        if not q:
            return catalog.head(top_n).copy()

        # Search across titles, synopsis, genres, cast and director.
        fields = ["title", "overview", "genres_text", "keywords_text", "cast_text", "director", "original_title"]
        mask = pd.Series(False, index=catalog.index)
        for field in fields:
            if field in catalog.columns:
                mask = mask | catalog[field].fillna("").astype(str).str.lower().str.contains(
                    re.escape(str(query).strip().lower()), regex=True, na=False
                )
        matches = catalog[mask].copy()
        if matches.empty:
            query_vector = self.vectorizer.transform([q])
            local_indices = catalog.index.to_numpy()
            scores = cosine_similarity(query_vector, self.matrix[local_indices]).ravel()
            order = np.argsort(scores)[::-1]
            picked = [local_indices[i] for i in order[:top_n] if scores[i] > 0]
            matches = self.df.loc[picked].copy() if picked else self.df.iloc[0:0].copy()
        return matches.sort_values(["popularity", "vote_count"], ascending=False).head(top_n)
