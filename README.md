# 🎬 Movie Recommendation System

A content-based movie recommendation engine built with Python and scikit-learn, using the MovieLens dataset. The system analyzes movie genres and user rating patterns to suggest similar movies, along with a full exploratory data analysis of viewing and rating trends.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📖 Overview

With the explosion of streaming platforms, recommendation systems have become essential for helping users discover content they'll actually enjoy. This project builds a **content-based movie recommendation system** using the MovieLens dataset, which contains movie metadata, genres, user ratings, and timestamps.

The core idea: movies are recommended based on how similar their **genres** are to a movie the user already likes, using **TF-IDF vectorization** and **cosine similarity** (via `NearestNeighbors`).

### Objectives
- Analyze movie and rating datasets
- Perform data cleaning, preprocessing, and visualization
- Build a content-based recommendation model
- Recommend similar movies based on a given title
- Explore rating patterns, genre popularity, and user activity trends

---

## 📂 Dataset

This project uses the **MovieLens dataset**, consisting of two files:

| File | Description |
|---|---|
| `movies.csv` | Movie ID, title (with release year), and pipe-separated genres |
| `ratings.csv` | User ID, movie ID, rating (0.5–5.0), and timestamp |

> Dataset source: [MovieLens Dataset on Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading, cleaning, and manipulation |
| `matplotlib` / `seaborn` | Data visualization and EDA |
| `scikit-learn` (`CountVectorizer`, `TfidfVectorizer`) | Converting genre text into numerical vectors |
| `scikit-learn` (`NearestNeighbors`, `cosine_similarity`) | Measuring similarity between movies |

---

## 🧹 Data Preprocessing

- Checked and handled missing values (empty genre fields filled with `''`)
- Checked for duplicate records in both datasets
- Merged `ratings.csv` and `movies.csv` on `movieId` for combined analysis
- Extracted release **year** from movie titles using regex

---

## 📊 Exploratory Data Analysis

The notebook explores:

- ⭐ **Top rated movies** — average rating per title
- 🔥 **Most popular movies** — number of ratings per title
- 📈 **Rating distribution** — histogram of all ratings
- 🎭 **Genre distribution** — frequency of each genre across the catalog
- 📉 **Movies released per year** — trend of movie releases over time
- 🧊 **User–movie ratings heatmap** — rating patterns across top active users and top rated movies

---

## 🤖 Recommendation Model

**Approach:** Content-Based Filtering on movie genres

1. Genres are vectorized using `TfidfVectorizer` (split on `|`)
2. A `NearestNeighbors` model is fit on the TF-IDF matrix using **cosine distance**
3. Given a movie title, the model retrieves the 10 nearest movies in genre space

```python
def recommend(movie_name):
    movie_name = movie_name.lower()
    idx = df_movies[df_movies['title'].str.lower() == movie_name].index

    if len(idx) == 0:
        print("Movie not found!")
        return

    idx = idx[0]
    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=11)

    print(f"Recommended Movies for '{df_movies.iloc[idx]['title']}'")
    for i in range(1, len(indices[0])):
        print(df_movies.iloc[indices[0][i]]['title'])
```

### Example Output

```
recommend("Toy Story (1995)")

🎬 Recommended Movies for 'Toy Story (1995)'

Antz (1998)
Toy Story 2 (1999)
Adventures of Rocky and Bullwinkle, The (2000)
Emperor's New Groove, The (2000)
...
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Project
1. Clone this repository
   ```bash
   git clone https://github.com/<your-username>/movie-recommendation-system.git
   cd movie-recommendation-system
   ```
2. Add `movies.csv` and `ratings.csv` to the project directory
3. Open and run the notebook
   ```bash
   jupyter notebook movie-recommendation-system.ipynb
   ```
4. Call the recommender:
   ```python
   recommend("Jumanji (1995)")
   ```

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── movie-recommendation-system.ipynb   # Main notebook (EDA + model)
├── movies.csv                          # Movie metadata (not included, download separately)
├── ratings.csv                         # User ratings (not included, download separately)
└── README.md                           # Project documentation
```

---

## 🔮 Future Improvements

- Incorporate **collaborative filtering** using user rating history
- Build a **hybrid recommender** combining content-based and collaborative approaches
- Add movie **tags, cast, and crew** as additional similarity features
- Deploy as an interactive **Streamlit / Flask web app**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [scikit-learn](https://scikit-learn.org/) documentation for modeling reference
