# MovieBot Viva Preparation

## Project summary (30–45 seconds)

"My project is MovieBot, a movie recommendation chatbot developed using Python and Streamlit. It uses the TMDB 5000 Movie Dataset from Kaggle. I preprocess movie metadata such as genres, keywords, overview, cast and director, combine these attributes into a text profile, and use TF-IDF vectorization with cosine similarity to recommend similar movies. The application also includes search, genre/rating/year filters, a session-based watchlist and dataset visualizations. It is a content-based recommendation system, so it can work without user-rating history or a paid AI API."

## Viva questions and answers

### 1. What is a recommendation system?
A software system that suggests items to a user based on item properties, user preferences, or patterns in user-item interactions.

### 2. What is content-based filtering?
It recommends items whose features resemble the features of an item the user likes. Movie features in this project include genres, synopsis, keywords, actors and director.

### 3. What dataset did you use?
The TMDB 5000 Movie Dataset on Kaggle. The project uses its movies CSV and credits CSV. The files are downloaded separately and placed in the `data/` directory.

### 4. Why did you use Python?
Python has libraries for data manipulation, machine learning and fast application development.

### 5. What is Pandas used for?
Reading CSV files, cleaning missing values, merging movie and credits records, and manipulating tabular data.

### 6. What is TF-IDF?
TF-IDF means Term Frequency–Inverse Document Frequency. It gives higher weight to terms that are relatively informative in a document and lower weight to terms that occur across many documents.

### 7. What is cosine similarity?
A similarity measure based on the cosine of the angle between two vectors. For normalized nonzero vectors, a higher cosine similarity generally means the feature vectors point in a more similar direction.

### 8. Why not use simple keyword matching only?
Keyword matching is easy but can miss related words and cannot rank partial matches well. TF-IDF represents text numerically and cosine similarity supports ranked comparisons.

### 9. Why did you use Streamlit?
Streamlit allows a Python developer to build an interactive web application without separately implementing a JavaScript frontend. Its chat components provide message containers and a chat input.

### 10. What is the difference between content-based and collaborative filtering?
Content-based filtering uses item attributes. Collaborative filtering uses user-item interactions or rating patterns across users. This project implements content-based filtering.

### 11. Is this generative AI?
No. The core is a classical machine-learning information retrieval/recommendation pipeline. It does not generate new text with an LLM.

### 12. How do you handle missing values?
The data loader fills missing textual fields with empty strings and converts numeric columns with safe defaults. Invalid release dates become missing years.

### 13. Why merge the movies and credits files?
The movies file contains movie metadata, while the credits file provides cast and crew information. Merging by movie ID enriches each movie profile.

### 14. What is a sparse matrix?
A matrix where most entries are zero. TF-IDF matrices are typically sparse because each document contains only a small subset of all possible vocabulary terms.

### 15. What is the time complexity?
Let N be the number of movies and V the vocabulary size. TF-IDF construction depends on the total text/token volume and vocabulary. A single query comparison against all N movie vectors costs roughly proportional to the number of nonzero features involved; exact cost depends on sparsity. Storing a full N×N similarity matrix would use O(N²) space, so this project computes similarity on demand instead.

### 16. What are the limitations?
Recommendations can be affected by incomplete metadata, synonymous terms, language differences and the absence of user-specific behaviour. Posters are not guaranteed by the CSV, and the watchlist is session-based rather than permanently stored.

### 17. How could you improve it?
Use a larger and cleaner dataset, add persistent SQLite storage, evaluate relevance with Precision@K or NDCG, incorporate user ratings for collaborative filtering, and optionally add an LLM for natural-language conversation.

### 18. How would you evaluate the recommender?
Create test queries or held-out relevance labels and measure Precision@K, Recall@K or NDCG@K. Also conduct qualitative checks to confirm that the recommendations are meaningfully similar.

### 19. What is a cold-start problem?
A recommendation challenge when there is little information about a new user or item. Content-based filtering can recommend a new movie if its metadata is available, but personalizing for a new user is still limited without preference data.

### 20. Why is a watchlist useful?
It gives users a way to collect movies they want to view. In this implementation it is stored in session state and may not persist permanently across sessions.

## Demonstration checklist

- [ ] Open the app successfully.
- [ ] Enter a known movie title, such as `movies like Inception`.
- [ ] Try a genre query, such as `romantic comedy`.
- [ ] Use the genre and rating filters.
- [ ] Save a recommendation to the watchlist.
- [ ] Open Dataset insights and explain the charts.
- [ ] Explain TF-IDF and cosine similarity on the whiteboard.
- [ ] Mention the limitations honestly.

## Suggested practical-file report structure

1. Title page
2. Abstract
3. Problem statement
4. Objectives
5. Hardware and software requirements
6. Dataset description
7. System architecture / flowchart
8. Algorithm and methodology
9. Implementation and screenshots
10. Test cases and results
11. Limitations and future scope
12. Conclusion
13. References
