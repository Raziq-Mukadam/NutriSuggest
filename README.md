# NutriSuggest

NutriSuggest is a PySpark and Streamlit recommendation-system microproject using the Food.com Recipes and User Interactions dataset. It combines recipe content (ingredients, tags, nutrition and preparation time) with Spark-computed review aggregates to rank recipes for a user's actual constraints.

## Dataset inspection and usage

The supplied `Dataset` folder was inspected before implementation. `RAW_recipes.csv` contains 231,637 raw rows and yields 231,636 usable recipe rows after null-ID/name removal and ID deduplication. `RAW_interactions.csv` contains 1,071,520 valid rating rows after rating-range validation and duplicate removal:

| File | Observed columns | Role |
|---|---|---|
| `RAW_recipes.csv` | `name`, `id`, `minutes`, `tags`, `nutrition`, `steps`, `ingredients`, ... | Primary recipe detail and content features |
| `RAW_interactions.csv` | `user_id`, `recipe_id`, `date`, `rating`, `review` | Recipe average ratings and review counts |
| `PP_recipes.csv`, `PP_users.csv` | Token/index features | Optional research/training assets; not required for the explainable UI |
| `interactions_train.csv`, `interactions_validation.csv`, `interactions_test.csv` | Indexed interaction splits | Optional collaborative-filtering experiments |

`src/data_loader.py` uses explicit Spark schemas, removes duplicate recipe IDs and duplicate user/recipe/date interactions, safely parses Food.com's Python-list strings, handles missing numeric values, and derives nutrition/diet labels. RAW files are read with Spark rather than loaded wholesale into Pandas.

## Architecture

1. **Ingestion:** Spark reads the two RAW CSV files with a schema.
2. **Cleaning:** duplicate IDs/interactions are removed; malformed list values become empty arrays; missing numeric values become safe zeroes.
3. **Feature preparation:** nutrition array positions become calories, fat, carbohydrates and protein; tags become approximate vegetarian, vegan, low-calorie, high-protein, low-fat and quick flags.
4. **Rating aggregation:** Spark computes average rating and review count and joins them to recipes.
5. **Recommendation:** hard constraints are applied in Spark. Ingredient overlap is counted, then a transparent score combines overlap, rating, review evidence and low-calorie preference.
6. **Presentation:** only small recommendation/aggregation results are converted to Pandas for Plotly or Streamlit rendering.

The current app intentionally uses content-based filtering. It is deterministic and explainable, so it never returns random recommendations. The indexed interaction files remain available for a future Spark ALS comparison.

## Installation (Windows)

1. Install Python 3.10+ and Java 8 or 11 (a current JDK is recommended).
2. From PowerShell:

```powershell
cd D:\NutriSuggest
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or invoke `.venv\Scripts\python.exe` directly.

## Dataset placement

Keep the downloaded CSV files in `D:\NutriSuggest\Dataset\`. The app requires:

- `RAW_recipes.csv`
- `RAW_interactions.csv`

The other supplied Food.com files may remain in the folder but are not required to launch the app.

## Run

```powershell
streamlit run app.py
```

The first launch parses the large raw files and builds Spark aggregates; subsequent reruns use Streamlit's resource cache. The browser opens at `http://localhost:8501`.

## Features

- Dashboard metrics for recipe count, rating, and approximate healthy categories.
- Deterministic recommendations from ingredients, dietary tags, calorie/time limits and minimum rating.
- Search by recipe name, ingredients and tags with time, calorie and rating filters.
- Expandable recipe cards showing ingredients, numbered instructions and nutrition.
- Spark-powered charts for common ingredients, popularity, ratings, cooking-time distribution and category calories.

Dietary labels are derived from Food.com tags and simple nutrition thresholds. They are approximate and do not guarantee allergy safety, medical suitability, vegetarian/vegan compliance, or serving-size accuracy.

## Abstract and objectives

**Abstract:** NutriSuggest demonstrates how distributed data processing can power a practical, explainable recipe recommendation system. Spark cleans and aggregates a large recipe/interaction dataset, while Streamlit turns those features into a responsive nutrition-focused interface.

**Objectives:** ingest a real CSV dataset; clean and transform semi-structured recipe fields; calculate rating and popularity analytics with Spark; provide deterministic content-based recommendations; and communicate the pipeline through an accessible dashboard.

## Results and conclusion

The system produces ranked recipes from the supplied data, not hard-coded or random examples. Its score is transparent and rewards ingredient matches, strong ratings, review evidence and lower calories while respecting user limits. Spark keeps the main data processing distributed and Pandas is restricted to small chart tables. The solution is suitable for a college demonstration; production use would add ingredient/allergen normalization, serving-aware nutrition validation and model evaluation against held-out interactions.

## Viva questions and answers

1. **Why PySpark?** It provides distributed CSV ingestion, transformations and aggregations suitable for large recipe and interaction data.
2. **Why content-based filtering?** Ingredients, tags and nutrition are available for every recipe and make recommendations explainable without requiring a logged-in user.
3. **How is the score calculated?** Hard filters enforce the request; ingredient overlap contributes 30 points per match, while rating, review evidence and low-calorie status add bounded points.
4. **How are healthy recipes identified?** Food.com tags and explicit nutrition thresholds provide approximate flags; the UI states their limitations.
5. **Why not convert all data to Pandas?** Pandas would increase memory pressure and bypass the Big Data requirement; only small result sets are converted for charts.
6. **How could this be extended?** Train Spark ALS on the supplied interaction splits and blend collaborative predictions with the content score.
7. **How are duplicates handled?** Recipes are deduplicated by `id`; interactions are deduplicated by `user_id`, `recipe_id` and `date`.
