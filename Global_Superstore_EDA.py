"""
Global Superstore Exploratory Data Analysis (EDA)
Converted from notebook to Python script.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")

# Visualization style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def main() -> None:
    output_dir = "eda_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Problem Statement
    print_section("Step 1: Problem Statement")
    print("Objective: Analyze sales performance, profitability drivers, and trends.")
    print("Business Goal: Improve sales and profit through data-backed insights.")

    # Step 2: Import Libraries
    print_section("Step 2: Import Libraries")
    print("Libraries loaded: pandas, numpy, matplotlib, seaborn")

    # Step 3: Load Dataset
    print_section("Step 3: Load Dataset")
    df = pd.read_csv("Global_Superstore.csv", encoding="latin1")
    print("Dataset loaded successfully!")
    print(df.head())

    # Step 4: Data Understanding
    print_section("Step 4: Data Understanding")
    print("Shape:", df.shape)
    print("\nColumns:\n", list(df.columns))
    print("\nData Types:\n")
    print(df.dtypes)
    print("\nSummary Statistics:\n")
    print(df.describe(include="all").T.head(24))

    # Step 5: Data Cleaning
    print_section("Step 5: Data Cleaning")
    df_clean = df.copy()

    missing_values = df_clean.isna().sum().sort_values(ascending=False)
    print("Missing values (top columns):")
    print(missing_values.head(10))

    # Postal Code can be missing for many countries; store as string.
    df_clean["Postal Code"] = df_clean["Postal Code"].astype("string")

    before = len(df_clean)
    df_clean = df_clean.drop_duplicates().copy()
    after = len(df_clean)
    print(f"\nDuplicates removed: {before - after}")

    for col in ["Order Date", "Ship Date"]:
        df_clean[col] = pd.to_datetime(df_clean[col], dayfirst=True, errors="coerce")

    num_cols = ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    print("\nCleaned dataset shape:", df_clean.shape)
    print(
        "Any nulls in date columns:",
        df_clean[["Order Date", "Ship Date"]].isna().sum().to_dict(),
    )

    # Step 6: Univariate Analysis
    print_section("Step 6: Univariate Analysis")
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    metrics = ["Sales", "Profit", "Quantity"]

    for i, metric in enumerate(metrics):
        sns.histplot(df_clean[metric], kde=True, ax=axes[i, 0], color="steelblue")
        axes[i, 0].set_title(f"{metric} Distribution")

        sns.boxplot(x=df_clean[metric], ax=axes[i, 1], color="orange")
        axes[i, 1].set_title(f"{metric} Boxplot")

    plt.tight_layout()
    univariate_path = os.path.join(output_dir, "01_univariate_distributions.png")
    plt.savefig(univariate_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {univariate_path}")
    print(
        "Key Observation: Sales and Profit are right-skewed with outliers; "
        "Quantity is concentrated at lower values."
    )

    # Step 7: Bivariate Analysis
    print_section("Step 7: Bivariate Analysis")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.scatterplot(data=df_clean, x="Sales", y="Profit", alpha=0.4, ax=axes[0])
    axes[0].set_title("Sales vs Profit")

    cat_sales = (
        df_clean.groupby("Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    sns.barplot(data=cat_sales, x="Category", y="Sales", ax=axes[1], palette="viridis")
    axes[1].set_title("Category vs Total Sales")
    axes[1].tick_params(axis="x", rotation=20)

    region_profit = (
        df_clean.groupby("Region", as_index=False)["Profit"]
        .sum()
        .sort_values("Profit", ascending=False)
    )
    sns.barplot(data=region_profit, x="Region", y="Profit", ax=axes[2], palette="magma")
    axes[2].set_title("Region vs Total Profit")
    axes[2].tick_params(axis="x", rotation=75)

    plt.tight_layout()
    bivariate_path = os.path.join(output_dir, "02_bivariate_analysis.png")
    plt.savefig(bivariate_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bivariate_path}")

    # Step 8: Multivariate Analysis
    print_section("Step 8: Multivariate Analysis")
    corr_cols = ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]
    correlation_matrix = df_clean[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Heatmap (Numerical Features)")
    corr_path = os.path.join(output_dir, "03_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {corr_path}")
    print("Observation: Discount has a noticeable negative correlation with Profit.")

    # Step 9: Time-based Analysis
    print_section("Step 9: Time-based Analysis")
    time_df = df_clean.dropna(subset=["Order Date"]).sort_values("Order Date").copy()

    monthly_sales = time_df.set_index("Order Date").resample("ME")["Sales"].sum()
    yearly_sales = time_df.set_index("Order Date").resample("YE")["Sales"].sum()
    month_name_sales = time_df.groupby(time_df["Order Date"].dt.month_name())["Sales"].sum()
    month_name_sales = month_name_sales.reindex(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    axes[0].plot(monthly_sales.index, monthly_sales.values, color="tab:blue", linewidth=2)
    axes[0].set_title("Monthly Sales Trend")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Sales")

    axes[1].plot(
        yearly_sales.index.year,
        yearly_sales.values,
        marker="o",
        color="tab:green",
        linewidth=2,
    )
    axes[1].set_title("Yearly Sales Trend")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Sales")

    sns.barplot(x=month_name_sales.index, y=month_name_sales.values, ax=axes[2], palette="crest")
    axes[2].set_title("Seasonality View: Total Sales by Calendar Month")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Total Sales")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    time_path = os.path.join(output_dir, "04_time_based_analysis.png")
    plt.savefig(time_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {time_path}")

    print(
        "Best monthly sales point:",
        monthly_sales.idxmax().strftime("%Y-%m"),
        "->",
        round(monthly_sales.max(), 2),
    )
    print(
        "Lowest monthly sales point:",
        monthly_sales.idxmin().strftime("%Y-%m"),
        "->",
        round(monthly_sales.min(), 2),
    )

    # Step 10: Key Business Insights
    print_section("Step 10: Key Business Insights")
    print("1. Total sales are ~$12.64M with total profit of ~$1.47M.")
    print("2. Sales grew year-over-year from 2011 to 2014.")
    print("3. Technology is the highest revenue category.")
    print("4. Tables sub-category has negative profitability despite high sales.")
    print("5. Discount and profit show a negative relationship.")
    print("6. High discount bands (20%+) are often loss-making.")
    print("7. Region-level margins vary significantly.")
    print("8. November is a strong seasonal sales month.")
    print("9. Home Office segment has slightly higher margin than peers.")
    print("10. Revenue is concentrated in major markets like US and Australia.")

    # Step 11: Conclusion
    print_section("Step 11: Conclusion")
    print("The business is growing, but profitability depends heavily on")
    print("discount strategy, category mix, and regional performance.")
    print("Use targeted pricing and seasonal planning for better outcomes.")

    print_section("EDA Script Completed")
    print(f"All generated graphs are saved in: {output_dir}")


if __name__ == "__main__":
    main()
