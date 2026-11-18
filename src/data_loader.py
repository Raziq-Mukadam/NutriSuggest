"""Spark data loading and cleaning for the Food.com dataset."""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

def find_dataset_file(data_dir: Path, filename: str) -> Path:
    """Find a required dataset file with a clear error if it is unavailable."""
    candidate = data_dir / filename
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing {filename}. Place the Food.com CSV files in {data_dir.resolve()}."
        )
    return candidate


def _literal_array(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _string_array(value: Any) -> list[str]:
    return [str(item).strip().lower() for item in _literal_array(value) if str(item).strip()]


def _float_array(value: Any) -> list[float]:
    result: list[float] = []
    for item in _literal_array(value):
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _schema_for_recipes() -> T.StructType:
    return T.StructType(
        [
            T.StructField("name", T.StringType(), True),
            T.StructField("id", T.LongType(), True),
            T.StructField("minutes", T.IntegerType(), True),
            T.StructField("contributor_id", T.LongType(), True),
            T.StructField("submitted", T.StringType(), True),
            T.StructField("tags", T.StringType(), True),
            T.StructField("nutrition", T.StringType(), True),
            T.StructField("n_steps", T.IntegerType(), True),
            T.StructField("steps", T.StringType(), True),
            T.StructField("description", T.StringType(), True),
            T.StructField("ingredients", T.StringType(), True),
            T.StructField("n_ingredients", T.IntegerType(), True),
        ]
    )


def _schema_for_interactions() -> T.StructType:
    return T.StructType(
        [
            T.StructField("user_id", T.LongType(), True),
            T.StructField("recipe_id", T.LongType(), True),
            T.StructField("date", T.StringType(), True),
            T.StructField("rating", T.DoubleType(), True),
            T.StructField("review", T.StringType(), True),
        ]
    )


def create_spark(app_name: str = "NutriSuggest") -> SparkSession:
    """Create a Windows-friendly local Spark session."""
    temp_dir = Path(__file__).resolve().parent.parent / ".spark_temp"
    temp_dir.mkdir(exist_ok=True)
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.local.dir", str(temp_dir))
        .getOrCreate()
    )


def load_recipes(spark: SparkSession, data_dir: Path) -> DataFrame:
    """Load, deduplicate, parse list fields, and derive nutrition columns."""
    path = find_dataset_file(data_dir, "RAW_recipes.csv")
    parse_strings = F.udf(_string_array, T.ArrayType(T.StringType()))
    parse_floats = F.udf(_float_array, T.ArrayType(T.DoubleType()))
    recipes = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .schema(_schema_for_recipes())
        .csv(str(path))
        .dropDuplicates(["id"])
        .filter(F.col("id").isNotNull() & F.col("name").isNotNull())
        .withColumn("minutes", F.coalesce(F.col("minutes"), F.lit(0)))
        .withColumn("tags_array", parse_strings("tags"))
        .withColumn("ingredients_array", parse_strings("ingredients"))
        .withColumn("steps_array", parse_strings("steps"))
        .withColumn("nutrition_array", parse_floats("nutrition"))
        .withColumn("calories", F.coalesce(F.element_at("nutrition_array", 1), F.lit(0.0)))
        .withColumn("fat", F.coalesce(F.element_at("nutrition_array", 2), F.lit(0.0)))
        .withColumn("saturated_fat", F.coalesce(F.element_at("nutrition_array", 3), F.lit(0.0)))
        .withColumn("cholesterol", F.coalesce(F.element_at("nutrition_array", 4), F.lit(0.0)))
        .withColumn("sodium", F.coalesce(F.element_at("nutrition_array", 5), F.lit(0.0)))
        .withColumn("carbohydrates", F.coalesce(F.element_at("nutrition_array", 6), F.lit(0.0)))
        .withColumn("protein", F.coalesce(F.element_at("nutrition_array", 7), F.lit(0.0)))
        .withColumn("is_vegetarian", F.array_contains("tags_array", "vegetarian"))
        .withColumn("is_vegan", F.array_contains("tags_array", "vegan"))
        .withColumn("is_quick", F.col("minutes").between(0, 30))
        .withColumn("is_low_calorie", F.col("calories") <= 400)
        .withColumn("is_high_protein", F.col("protein") >= 20)
        .withColumn("is_low_fat", F.col("fat") <= 15)
    )
    return recipes


def load_interactions(spark: SparkSession, data_dir: Path) -> DataFrame:
    """Load ratings and retain the fields needed for recipe-level aggregates."""
    path = find_dataset_file(data_dir, "RAW_interactions.csv")
    return (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .schema(_schema_for_interactions())
        .csv(str(path))
        .filter(F.col("recipe_id").isNotNull() & F.col("rating").between(1, 5))
        .dropDuplicates(["user_id", "recipe_id", "date"])
    )


def recipe_rating_aggregates(interactions: DataFrame) -> DataFrame:
    """Compute review count and mean rating using Spark."""
    return interactions.groupBy("recipe_id").agg(
        F.round(F.avg("rating"), 2).alias("average_rating"),
        F.count("*").alias("review_count"),
    )


def load_and_prepare(spark: SparkSession, data_dir: Path) -> DataFrame:
    """Load both raw files and join rating features to recipes."""
    recipes = load_recipes(spark, data_dir)
    ratings = recipe_rating_aggregates(load_interactions(spark, data_dir))
    return (
        recipes.join(ratings, recipes.id == ratings.recipe_id, "left")
        .drop("recipe_id")
        .withColumn("average_rating", F.coalesce("average_rating", F.lit(0.0)))
        .withColumn("review_count", F.coalesce("review_count", F.lit(0)))
        .cache()
    )


def inspect_csv(path: Path, sample_rows: int = 3) -> dict[str, Any]:
    """Return lightweight schema/sample information for the setup report."""
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        rows = [next(reader) for _ in range(sample_rows + 1)]
    return {"columns": rows[0], "samples": rows[1:]}
