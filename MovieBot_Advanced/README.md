# MovieBot — Advanced Movie Recommendation Chatbot

A B.Tech CSE practical project built with Python, Streamlit, Pandas, NumPy and scikit-learn. It uses content-based filtering to recommend movies from the Kaggle TMDB 5000 Movie Dataset.

## Features

- Conversational chat interface with chat history
- Movie-to-movie similarity recommendations
- Text search for genres, plot terms, actors and directors
- Genre, rating and release-year filters
- Movie details, synopsis, cast and director
- Session-based watchlist
- Dataset insights and charts
- No paid API key required

## 1. Requirements

- Python 3.10 or newer recommended
- VS Code or another Python editor
- Internet access to download the dataset and install packages

## 2. Download the dataset

Open the Kaggle dataset page:

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Download and extract the archive. Copy these files into the `data` folder:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Expected layout:

```text
MovieBot_Advanced/
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
└── data/
    ├── tmdb_5000_movies.csv
    ├── tmdb_5000_credits.csv
    └── README.txt
```

The dataset is not included in this source ZIP; download it from Kaggle yourself and comply with the dataset's terms.

## 3. Create a virtual environment

### Windows (PowerShell)

Open a terminal in the `MovieBot_Advanced` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the app

From the project folder:

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## 5. Try these prompts

- `movies like Inception`
- `romantic comedy`
- `mind-bending science fiction`
- `films with Leonardo DiCaprio`
- `Christopher Nolan`
- `action adventure`
- `friendship and college life`

The engine uses metadata and text similarity. It does not understand natural language like a large language model; unusual phrasing may produce weak matches.

## How the recommendation engine works

1. **Data preprocessing:** Parse JSON-like genres, keywords, cast and crew columns from the Kaggle CSV files.
2. **Feature engineering:** Combine title, genres, keywords, synopsis, cast, director and tagline into a text profile.
3. **TF-IDF:** Transform each movie profile into a sparse numerical vector.
4. **Cosine similarity:** Compare a selected movie or user query against movie vectors.
5. **Ranking:** Return the top matching movies, excluding the selected title when recommending similar movies.
6. **Interface:** Streamlit displays chat messages, movie cards, filters, a watchlist and dataset insights.

## Known limitations

- The supplied Kaggle metadata does not guarantee poster URLs. The UI displays a simple fallback when no poster URL is available.
- The watchlist and chat history live in Streamlit session state and are not a permanent account/database.
- Recommendations depend on metadata quality and keyword overlap.
- This is a content-based recommender, not collaborative filtering and not a generative AI chatbot.
- This source has not been run against your local copy of the Kaggle CSVs; test it after downloading the dataset.

## Suggested improvements

- Use TMDB's official API for poster images and extra movie details, subject to its terms.
- Persist watchlists in SQLite.
- Add tests for data parsing and ranking.
- Add collaborative filtering if user-rating data becomes available.
- Add a language-model API for richer conversational understanding, with an API key stored securely.
