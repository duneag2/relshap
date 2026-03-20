import os
import duckdb
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="make_movielens20m"
)

parser.add_argument(
    "--base-dir",
    type=str,
    required=True,
    help="data directory",
)

parser.add_argument(
    "--db",
    type=str,
    required=True,
    help="duckdb file",
)

parser.add_argument(
    "--query",
    type=str,
    required=True,
    help="query file",
)

parser.add_argument(
    "--flattened",
    type=str,
    required=True,
    help="flattened csv file",
)

args = parser.parse_args()

CSV_DIR = os.path.join(args.base_dir, "movielens20m")
# CSV_DIR = os.path.join('/Users/seungeun/nyu/relshap2026/copy_safe_feb92pm/dataset/movielens20m', "movielens20m")
DB_PATH = os.path.join(args.base_dir, args.db)
SQL_PATH = os.path.join(args.base_dir, args.query)
CSV_PATH = os.path.join(args.base_dir, args.flattened)


paths = {
    "rating": os.path.join(CSV_DIR, "rating.csv"),
    "movie": os.path.join(CSV_DIR, "movie.csv"),
    "genome_scores": os.path.join(CSV_DIR, "genome_scores.csv"),
}

movie_raw = pd.read_csv(paths["movie"])

def normalize_genres(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)) or (isinstance(s, str) and s.strip() == ""):
        return ""
    parts = [g.strip() for g in str(s).split("|") if g.strip()]
    parts.sort()
    return "|".join(parts)

movie_raw["genres_norm"] = movie_raw["genres"].apply(normalize_genres)
movie_raw["genres_id"], _ = pd.factorize(movie_raw["genres_norm"])

movie_df = pd.DataFrame({
    "movieId": movie_raw["movieId"].astype("int64"),
    "genres": movie_raw["genres_id"].astype("int64"),
})

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
con = duckdb.connect(database=DB_PATH)

con.execute("DROP TABLE IF EXISTS genome_scores;")
con.execute("DROP TABLE IF EXISTS rating;")
con.execute("DROP TABLE IF EXISTS movie;")

con.execute("""
CREATE TABLE movie (
    movieId INTEGER PRIMARY KEY,
    genres  INTEGER
);
""")

con.register("movie_df", movie_df)

con.execute("""
INSERT INTO movie
SELECT
    CAST(movieId AS INTEGER),
    CAST(genres  AS INTEGER)
FROM movie_df;
""")

con.execute("""
CREATE TABLE rating (
    userId  INTEGER,
    movieId INTEGER,
    rating  DOUBLE,
    PRIMARY KEY (userId, movieId),
    FOREIGN KEY (movieId) REFERENCES movie(movieId)
);
""")

con.execute(f"""
INSERT INTO rating
SELECT
    CAST(userId  AS INTEGER),
    CAST(movieId AS INTEGER),
    CAST(rating  AS DOUBLE)
FROM read_csv_auto('{paths["rating"]}', header=True);
""")

con.execute("""
CREATE TABLE genome_scores (
    movieId   INTEGER,
    tagId     INTEGER,
    relevance DOUBLE,
    PRIMARY KEY (movieId, tagId),
    FOREIGN KEY (movieId) REFERENCES movie(movieId)
);
""")

con.execute(f"""
INSERT INTO genome_scores
SELECT
    CAST(movieId   AS INTEGER),
    CAST(tagId     AS INTEGER),
    CAST(relevance AS DOUBLE)
FROM read_csv_auto('{paths["genome_scores"]}', header=True);
""")

con.close()

with open(SQL_PATH, "r") as f:
    query = f.read()

con = duckdb.connect(database=DB_PATH, read_only=True)

df = con.execute(query).df()
# print(df)

# rating 기준 median
rating_median = df["rating"].median()

# median 이상 = 1, 미만 = 0
df["rating"] = (df["rating"] >= rating_median).astype(int)

df.to_csv(CSV_PATH, index=False)


con.close()
