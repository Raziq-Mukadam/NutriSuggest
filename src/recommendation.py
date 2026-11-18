"""Content-based recipe recommendation implemented with Spark filters and scoring."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from .preprocessing import apply_search


def recommend(
    recipes: DataFrame,
    ingredients: list[str],
    categories: list[str],
    max_calories: float,
    max_minutes: int,
    min_rating: float,
    limit: int = 24,
) -> DataFrame:
    """Score actual dataset rows by ingredient overlap, preference matches, and rating."""
    health_filters = {
        "low-calorie": "is_low_calorie",
        "high-protein": "is_high_protein",
        "vegetarian": "is_vegetarian",
        "vegan": "is_vegan",
        "low-fat": "is_low_fat",
        "quick": "is_quick",
    }
    tag_categories = [category for category in categories if category not in health_filters]
    filtered = apply_search(
        recipes,
        max_minutes=max_minutes,
        max_calories=max_calories,
        min_rating=min_rating,
        categories=tag_categories,
    )
    for category in categories:
        flag_column = health_filters.get(category)
        if flag_column:
            filtered = filtered.filter(F.col(flag_column))
    cleaned = [item.strip().lower() for item in ingredients if item.strip()]
    if cleaned:
        overlap = F.aggregate(
            F.transform(
                F.array(*[F.lit(item) for item in cleaned]),
                lambda item: F.when(
                    F.exists("ingredients_array", lambda ingredient: ingredient.contains(item)),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            ),
            F.lit(0),
            lambda total, item: total + item,
        )
        filtered = filtered.withColumn("ingredient_matches", overlap).filter(
            F.col("ingredient_matches") > 0
        )
    else:
        filtered = filtered.withColumn("ingredient_matches", F.lit(0))
    return (
        filtered.withColumn(
            "recommendation_score",
            F.round(
                F.col("ingredient_matches") * 30
                + F.least(F.col("average_rating") * 10, F.lit(50))
                + F.least(F.col("review_count") / 100, F.lit(10))
                + F.when(F.col("is_low_calorie"), 5).otherwise(0),
                2,
            ),
        )
        .orderBy(F.desc("recommendation_score"), F.desc("average_rating"), F.asc("minutes"))
        .limit(limit)
    )
