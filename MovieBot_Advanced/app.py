import streamlit as st
import pandas as pd
from pathlib import Path
from recommender import MovieRecommender, load_movies

st.set_page_config(
    page_title="MovieBot | Discover your next film",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {
  padding: 1.5rem 1.7rem; border-radius: 20px; margin-bottom: 1.1rem;
  background: linear-gradient(120deg, rgba(76,29,149,.42), rgba(30,41,59,.82));
  border: 1px solid rgba(167,139,250,.25);
}
.hero h1 {margin: 0; font-size: 2.25rem;}
.hero p {margin: .5rem 0 0 0; opacity: .83;}
.movie-card {
  padding: 1rem; border-radius: 16px; border: 1px solid rgba(140,140,140,.25);
  height: 100%; background: rgba(120,120,120,.04);
}
.movie-title {font-size: 1.05rem; font-weight: 700; margin-bottom: .35rem;}
.movie-meta {font-size: .84rem; opacity: .78; margin-bottom: .55rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Preparing the recommendation engine…")
def get_engine():
    df = load_movies(DATA_DIR)
    return MovieRecommender(df)

try:
    engine = get_engine()
except Exception as exc:
    st.error("Movie dataset could not be loaded.")
    st.markdown(
        "Download the **TMDB 5000 Movie Dataset** from Kaggle and place "
        "`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` inside the `data/` folder. "
        "See `README.md` for the exact steps."
    )
    st.exception(exc)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Welcome to MovieBot! 🍿 Tell me what you feel like watching, "
            "name a movie you love, or try a genre such as comedy, romance, "
            "thriller, science fiction, or adventure."
        ),
        "results": [],
    }]
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []

with st.sidebar:
    st.markdown("## 🎞️ MovieBot")
    st.caption("A content-based movie discovery assistant")
    page = st.radio("Explore", ["Chatbot", "Discover", "Watchlist", "Dataset insights"])
    st.divider()
    st.markdown("### Quick filters")
    all_genres = engine.all_genres()
    selected_genres = st.multiselect("Genres", all_genres, placeholder="Any genre")
    min_rating = st.slider("Minimum TMDB rating", 0.0, 9.0, 0.0, 0.5)
    year_range = st.slider("Release year", 1900, 2025, (2000, 2025))
    st.caption("Filters are used on the Discover page and when refining chatbot recommendations.")

def movie_card(movie, key_prefix):
    title = str(movie.get("title", "Untitled"))
    overview = str(movie.get("overview", "") or "No overview is available.")
    genres = str(movie.get("genres_text", "") or "Genre not listed")
    year = movie.get("release_year", None)
    year_text = str(int(year)) if pd.notna(year) else "Year unknown"
    rating = movie.get("vote_average", None)
    rating_text = f"⭐ {float(rating):.1f}/10" if pd.notna(rating) else "Rating unavailable"

    with st.container(border=True):
        poster = movie.get("poster_url", "")
        if isinstance(poster, str) and poster.startswith("http"):
            st.image(poster, use_container_width=True)
        else:
            st.markdown("### 🎬")
        st.markdown(f"**{title}**")
        st.caption(f"{year_text} · {rating_text}")
        st.caption(genres)
        with st.expander("Synopsis and details"):
            st.write(overview)
            cast = movie.get("cast_text", "")
            director = movie.get("director", "")
            if director:
                st.write(f"**Director:** {director}")
            if cast:
                st.write(f"**Cast:** {cast}")
        if st.button(
            "♥ Save to watchlist",
            key=f"{key_prefix}_{movie.get('movie_id', title)}",
            use_container_width=True,
        ):
            item = {
                "movie_id": int(movie["movie_id"]) if pd.notna(movie.get("movie_id")) else title,
                "title": title,
                "release_year": year_text,
                "vote_average": rating_text,
                "genres_text": genres,
                "overview": overview,
                "poster_url": poster if isinstance(poster, str) else "",
            }
            if not any(x["movie_id"] == item["movie_id"] for x in st.session_state.watchlist):
                st.session_state.watchlist.append(item)
                st.toast(f"Saved {title} to your watchlist!")
            else:
                st.toast("That movie is already in your watchlist.")

def render_results(results, key_prefix):
    if results is None or len(results) == 0:
        st.info("No close matches found. Try a movie title or a broader genre, such as drama or comedy.")
        return
    st.markdown(f"#### 🍿 Recommended for you ({len(results)})")
    cols = st.columns(3)
    for idx, (_, movie) in enumerate(results.iterrows()):
        with cols[idx % 3]:
            movie_card(movie, f"{key_prefix}_{idx}")

def apply_sidebar_filters(df):
    out = df.copy()
    if selected_genres:
        out = out[out["genres_list"].apply(lambda gs: any(g in gs for g in selected_genres))]
    if "vote_average" in out.columns:
        out = out[pd.to_numeric(out["vote_average"], errors="coerce").fillna(0) >= min_rating]
    if "release_year" in out.columns:
        out = out[
            pd.to_numeric(out["release_year"], errors="coerce").between(year_range[0], year_range[1])
            | out["release_year"].isna()
        ]
    return out

st.markdown(
    '<div class="hero"><h1>🎬 Find your next favourite movie</h1>'
    '<p>Explore stories, discover similar films, and build your personal watchlist.</p></div>',
    unsafe_allow_html=True,
)

if page == "Chatbot":
    st.markdown("### Chat with MovieBot")
    st.caption("Examples: “movies like Inception”, “mind-bending sci-fi”, “romantic comedy”, “films directed by Christopher Nolan”")
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("results"):
                render_results(engine.frame_by_ids(message["results"]), f"chat_{i}")

    prompt = st.chat_input("Tell me what kind of movie you want…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "results": []})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Finding movies that match your taste…"):
                results, explanation = engine.recommend_from_text(prompt, top_n=6)
            st.markdown(explanation)
            if not results.empty:
                render_results(results, f"new_{len(st.session_state.messages)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": explanation,
                "results": results["movie_id"].tolist() if not results.empty else [],
            })
            st.session_state.last_results = results["movie_id"].tolist() if not results.empty else []

    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat cleared. What would you like to watch next?",
            "results": [],
        }]
        st.rerun()

elif page == "Discover":
    st.markdown("### Browse the movie catalogue")
    query = st.text_input("Search titles, actors, directors, or plot keywords", placeholder="e.g. space, Leonardo DiCaprio, friendship")
    catalog = apply_sidebar_filters(engine.df)
    if query.strip():
        found = engine.search_catalog(query, catalog=catalog, top_n=60)
    else:
        found = catalog.sort_values(
            ["popularity", "vote_count"], ascending=False
        ).head(60)
    st.caption(f"Showing {len(found)} movies from the filtered catalogue.")
    if found.empty:
        st.warning("No movies match those filters. Try lowering the rating or widening the year range.")
    else:
        render_results(found, "discover")

elif page == "Watchlist":
    st.markdown("### ♥ Your watchlist")
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Save a movie from Chatbot or Discover to add it here.")
    else:
        st.caption(f"{len(st.session_state.watchlist)} saved movie(s). The watchlist is kept in this browser session.")
        if st.button("Clear watchlist"):
            st.session_state.watchlist = []
            st.rerun()
        render_results(pd.DataFrame(st.session_state.watchlist), "watchlist")

else:
    st.markdown("### 📊 Dataset insights")
    df = engine.df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{len(df):,}")
    c2.metric("Genres", len(engine.all_genres()))
    c3.metric("Average rating", f"{pd.to_numeric(df['vote_average'], errors='coerce').mean():.2f}/10")
    c4.metric("With synopsis", f"{df['overview'].fillna('').str.len().gt(0).mean():.0%}")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Most common genres")
        genre_counts = (
            df.explode("genres_list")["genres_list"]
            .replace("", pd.NA).dropna().value_counts().head(12)
        )
        st.bar_chart(genre_counts)
    with right:
        st.markdown("#### Rating distribution")
        ratings = pd.to_numeric(df["vote_average"], errors="coerce").dropna()
        st.bar_chart(ratings.round().value_counts().sort_index())
    st.markdown("#### Sample records")
    display_cols = [c for c in ["title", "release_year", "genres_text", "vote_average", "popularity"] if c in df.columns]
    st.dataframe(df[display_cols].head(25), use_container_width=True, hide_index=True)

st.divider()
st.caption("MovieBot is a student project. Recommendations are based on available metadata and similarity scores, not guaranteed personal preferences.")
