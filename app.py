"""NutriSuggest: personalized healthy recipe recommendations with PySpark."""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st
from py4j.protocol import Py4JJavaError
from pyspark.sql import functions as F

from src.analytics import (
    category_calories,
    cooking_time_distribution,
    popular,
    rating_distribution,
    summary,
    top_rated,
)
from src.data_loader import create_spark, load_and_prepare
from src.recommendation import recommend
from src.preprocessing import apply_search, ingredient_frequency

st.set_page_config(page_title="NutriSuggest", page_icon="🥗", layout="wide")

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: #ffffff; color: #262730; }
    [data-testid="stHeader"] { background: #ffffff; }
    .main .block-container { color: #262730; }
    .main h1, .main h2, .main h3, .main h4, .main p,
    .main label, .main [data-testid="stMarkdownContainer"] { color: #262730; }
    [data-testid="stSidebar"] { background: #f0f2f6; }
    [data-testid="stSidebar"] * { color: #262730 !important; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [role="radiogroup"] label { color: #262730 !important; }
    [data-baseweb="popover"] *, [role="listbox"] *, [role="option"] { color: #262730 !important; }
    [data-baseweb="select"] input,
    [data-baseweb="select"] [data-baseweb="input"] { color: #262730 !important; }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] textarea { color: #262730 !important; background: #ffffff !important; }
    [data-testid="stTextInput"] input::placeholder { color: #6b8174 !important; opacity: 1; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] { color: #262730 !important; }
    [data-testid="stExpander"] { color: #262730; background: #ffffff; border-color: #dfeee2; }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p { color: #262730 !important; }
    .hero { padding: 1.5rem 2rem; border-radius: 22px; background: linear-gradient(120deg,#12372a,#4c956c); color: white; margin-bottom: 1.2rem; }
    .hero h1, .hero p { color: white !important; }
    .hero h1 { margin: 0; font-size: 2.7rem; }
    .hero p { margin: .4rem 0 0; font-size: 1.05rem; opacity: .9; }
    .recipe-card { padding: 1rem 1.2rem; border: 1px solid #dfeee2; border-radius: 16px; background: white; color: #262730; margin-bottom: .8rem; box-shadow: 0 3px 12px #16351b12; }
    .recipe-card h3, .recipe-card p { color: #262730 !important; }
    .pill { display: inline-block; padding: .2rem .55rem; margin: .15rem; background: #e7f5e9; color: #23613c; border-radius: 99px; font-size: .78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "Dataset"


@st.cache_resource(show_spinner=False)
def get_data(data_dir: str):
    spark = create_spark()
    return spark, load_and_prepare(spark, Path(data_dir))


def render_recipe(row) -> None:
    nutrition = (
        f"🔥 {row.calories:.0f} kcal  ·  💪 {row.protein:.1f}g protein  ·  "
        f"🥑 {row.fat:.1f}g fat  ·  🌾 {row.carbohydrates:.1f}g carbs"
    )
    st.markdown(
        f'<div class="recipe-card"><h3>🍽️ {row.name.title()}</h3>'
        f'<span class="pill">⭐ {row.average_rating:.2f}</span>'
        f'<span class="pill">💬 {row.review_count:,} reviews</span>'
        f'<span class="pill">⏱️ {row.minutes} min</span><p>{nutrition}</p></div>',
        unsafe_allow_html=True,
    )
    with st.expander("View ingredients and instructions"):
        st.markdown("**Ingredients**")
        st.write(", ".join(row.ingredients_array).capitalize())
        st.markdown("**Instructions**")
        for index, step in enumerate(row.steps_array, 1):
            st.write(f"{index}. {step.capitalize()}")


def main() -> None:
    st.markdown(
        '<div class="hero"><h1>🥗 NutriSuggest</h1><p>Smart, data-driven recipes for the ingredients and lifestyle you already have.</p></div>',
        unsafe_allow_html=True,
    )
    if not DATA_DIR.exists():
        st.error(f"Dataset folder not found: {DATA_DIR}")
        st.stop()
    try:
        spark, recipes = get_data(str(DATA_DIR))
        stats = summary(recipes)
    except (FileNotFoundError, OSError, Py4JJavaError) as exc:
        st.error(f"Unable to load the dataset with Spark: {exc}")
        st.info("Confirm Java 8+ and the RAW_recipes.csv / RAW_interactions.csv files are available.")
        st.stop()

    with st.sidebar:
        st.header("Navigate")
        page = st.radio("Section", ["Dashboard", "Recommend", "Search", "Analytics"])
        st.divider()
        st.caption("Dataset: Food.com RAW files")
        st.caption("All classifications are approximate and should be checked against recipe labels.")

    if page == "Dashboard":
        st.subheader("Your nutrition snapshot")
        cols = st.columns(5)
        cols[0].metric("Recipes", f"{int(stats['recipes']):,}")
        cols[1].metric("Average rating", f"{stats['rating']:.2f}/5")
        cols[2].metric("Low calorie", f"{int(stats['low_calorie']):,}")
        cols[3].metric("Vegetarian", f"{int(stats['vegetarian']):,}")
        cols[4].metric("Vegan", f"{int(stats['vegan']):,}")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.bar(ingredient_frequency(recipes).toPandas(), x="count", y="ingredient", orientation="h", title="Most common ingredients"), use_container_width=True)
        with right:
            st.plotly_chart(px.pie(cooking_time_distribution(recipes).toPandas(), names="time_band", values="count", title="Recipe distribution by cooking time"), use_container_width=True)
        st.caption("The dashboard uses Spark transformations and only converts small, aggregated result sets to Pandas for Plotly.")
    elif page == "Recommend":
        st.subheader("Find your next healthy favorite")
        c1, c2 = st.columns(2)
        with c1:
            ingredients = st.text_input("Ingredients you have", placeholder="chicken, tomato, garlic")
            max_calories = st.slider("Maximum calories", 50, 2000, 600, 25)
            max_minutes = st.slider("Maximum cooking time (minutes)", 5, 600, 90, 5)
        with c2:
            categories = st.multiselect(
                "Healthy and dietary filters",
                ["low-calorie", "high-protein", "vegetarian", "vegan", "low-fat", "quick", "healthy"],
            )
            min_rating = st.slider("Minimum rating", 0.0, 5.0, 3.5, 0.5)
            recommendation_count = st.slider("Number of recommendations", 5, 10, 10)
        if st.button("✨ Generate recommendations", type="primary"):
            with st.spinner("Scoring recipes with Spark..."):
                results = recommend(
                    recipes,
                    ingredients.split(","),
                    categories,
                    max_calories,
                    max_minutes,
                    min_rating,
                    limit=recommendation_count,
                ).collect()
            st.success(f"Found {len(results)} recipes ranked from the real Food.com dataset.")
            if results:
                for row in results:
                    render_recipe(row)
            else:
                st.warning("No matching recipes. Try fewer ingredients or wider limits.")
    elif page == "Search":
        st.subheader("Search the recipe library")
        query = st.text_input("Search names, ingredients, or tags", placeholder="pasta, tomato")
        c1, c2, c3 = st.columns(3)
        max_minutes = c1.number_input("Max minutes", min_value=1, value=180)
        max_calories = c2.number_input("Max calories", min_value=1, value=1000)
        min_rating = c3.slider("Min rating", 0.0, 5.0, 0.0, 0.5)
        results = apply_search(recipes, query, max_minutes, max_calories, min_rating).orderBy(F.desc("average_rating")).limit(30).collect()
        st.caption(f"Showing up to 30 of the best matches ({len(results)} loaded).")
        for row in results:
            render_recipe(row)
    else:
        st.subheader("Big Data analytics")
        tab1, tab2, tab3 = st.tabs(["Popularity", "Ratings", "Nutrition"])
        with tab1:
            st.plotly_chart(px.bar(popular(recipes).toPandas(), x="review_count", y="name", orientation="h", title="Most popular recipes"), use_container_width=True)
            st.dataframe(top_rated(recipes).toPandas(), use_container_width=True, hide_index=True)
        with tab2:
            st.plotly_chart(px.bar(rating_distribution(recipes).toPandas(), x="rating", y="count", title="Average rating distribution"), use_container_width=True)
        with tab3:
            st.plotly_chart(px.bar(category_calories(recipes).toPandas(), x="category", y="average_calories", title="Average calories by category"), use_container_width=True)


if __name__ == "__main__":
    main()
