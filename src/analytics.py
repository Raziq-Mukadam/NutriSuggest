"""Spark-powered analytical datasets used by the Streamlit dashboard."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F


def summary(recipes: DataFrame) -> dict[str, float]:
    row = recipes.agg(
        F.count("*").alias("recipes"),
        F.round(F.avg("average_rating"), 2).alias("rating"),
        F.sum(F.col("is_low_calorie").cast("int")).alias("low_calorie"),
        F.sum(F.col("is_vegetarian").cast("int")).alias("vegetarian"),
        F.sum(F.col("is_vegan").cast("int")).alias("vegan"),
    ).first()
    return {key: (value or 0) for key, value in row.asDict().items()}


def top_rated(recipes: DataFrame, limit: int = 10) -> DataFrame:
    return recipes.filter(F.col("review_count") >= 5).orderBy(
        F.desc("average_rating"), F.desc("review_count")
    ).select("name", "average_rating", "review_count").limit(limit)


def popular(recipes: DataFrame, limit: int = 10) -> DataFrame:
    return recipes.orderBy(F.desc("review_count"), F.desc("average_rating")).select(
        "name", "average_rating", "review_count"
    ).limit(limit)


def rating_distribution(recipes: DataFrame) -> DataFrame:
    return recipes.select(F.round("average_rating", 0).alias("rating")).groupBy("rating").count().orderBy("rating")


def cooking_time_distribution(recipes: DataFrame) -> DataFrame:
    return recipes.select(
        F.when(F.col("minutes") <= 30, "0-30 min")
        .when(F.col("minutes") <= 60, "31-60 min")
        .when(F.col("minutes") <= 120, "61-120 min")
        .otherwise("120+ min")
        .alias("time_band")
    ).groupBy("time_band").count()


def category_calories(recipes: DataFrame) -> DataFrame:
    return (
        recipes.select(F.explode("tags_array").alias("category"), "calories")
        .filter(
            F.col("category").isin(
                "main-dish", "desserts", "breakfast", "vegetables", "healthy"
            )
        )
        .groupBy("category")
        .agg(F.round(F.avg("calories"), 1).alias("average_calories"))
    )
