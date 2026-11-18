"""Reusable Spark transformations for recipe search and dietary labelling."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F


def apply_search(
    recipes: DataFrame,
    query: str = "",
    max_minutes: int | None = None,
    max_calories: float | None = None,
    min_rating: float = 0.0,
    categories: list[str] | None = None,
) -> DataFrame:
    result = recipes
    if query.strip():
        terms = [term.strip().lower() for term in query.split(",") if term.strip()]
        for term in terms:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            result = result.filter(
                F.lower(
                    F.concat_ws(
                        " ",
                        F.col("name"),
                        F.concat_ws(" ", "ingredients_array"),
                        F.concat_ws(" ", "tags_array"),
                    )
                ).contains(escaped)
            )
    if max_minutes is not None:
        result = result.filter(F.col("minutes") <= max_minutes)
    if max_calories is not None:
        result = result.filter(F.col("calories") <= max_calories)
    if min_rating:
        result = result.filter(F.col("average_rating") >= min_rating)
    for category in categories or []:
        result = result.filter(F.array_contains("tags_array", category.lower()))
    return result


def ingredient_frequency(recipes: DataFrame, limit: int = 20) -> DataFrame:
    return (
        recipes.select(F.explode("ingredients_array").alias("ingredient"))
        .filter(F.length("ingredient") > 1)
        .groupBy("ingredient")
        .count()
        .orderBy(F.desc("count"))
        .limit(limit)
    )
