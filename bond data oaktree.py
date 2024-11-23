import requests
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import os
import numpy as np

# Your FRED API key
API_KEY = "236231523c688c3f3ddaf5b03a1fce62"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_yield_data(series_id):
    """
    Fetch data from FRED API for a given series ID.
    """
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json"
    }
    try:
        print(f"Fetching data for {series_id}...")
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        observations = data.get("observations", [])
        df = pd.DataFrame(observations)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["date", "value"]]
    except Exception as e:
        print(f"Error fetching data for {series_id}: {e}")
        return pd.DataFrame()

# List of Treasury yield series IDs on FRED
tenor_series_ids = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1",
    "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"
]

start_date = pd.to_datetime("2023-01-01").date()
end_date = pd.to_datetime("2023-12-31").date()

data_frames = {}
for series_id in tenor_series_ids:
    df = fetch_yield_data(series_id)
    if not df.empty:
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.set_index("date")
        df.rename(columns={"value": series_id}, inplace=True)
        data_frames[series_id] = df

if data_frames:
    combined_df = pd.concat(data_frames.values(), axis=1)
    combined_df = combined_df.apply(pd.to_numeric, errors="coerce")
    treasury_yield_mapping = {
        "DGS1MO": "UST1MO", "DGS3MO": "UST3MO", "DGS6MO": "UST6MO",
        "DGS1": "UST1YR", "DGS2": "UST2YR", "DGS3": "UST3YR",
        "DGS5": "UST5YR", "DGS7": "UST7YR", "DGS10": "UST10YR",
        "DGS20": "UST20YR", "DGS30": "UST30YR"
    }
    combined_df.rename(columns=treasury_yield_mapping, inplace=True)
else:
    print("No data fetched. Please check the API key or series IDs.")

# Load the bond data
bond_data_path = r"C:\Users\valim\Downloads\Part 1. bonds_yields.xlsx"
bond_df = pd.read_excel(bond_data_path)

target_date = pd.to_datetime("2023-12-29").date()
if target_date in combined_df.index:
    treasury_curve = combined_df.loc[target_date].dropna()
else:
    most_recent_date = max(combined_df.index)
    treasury_curve = combined_df.loc[most_recent_date].dropna()

maturities = {
    "UST1MO": 1 / 12, "UST3MO": 3 / 12, "UST6MO": 6 / 12, "UST1YR": 1,
    "UST2YR": 2, "UST3YR": 3, "UST5YR": 5, "UST7YR": 7, "UST10YR": 10,
    "UST20YR": 20, "UST30YR": 30
}

treasury_data = pd.DataFrame({
    "Maturity": maturities.values(),
    "Yield": [treasury_curve[label] for label in maturities.keys() if label in treasury_curve.index]
}).dropna().sort_values(by="Maturity")

def interpolate_treasury_yield(wal, treasury_data):
    if wal <= treasury_data["Maturity"].min():
        return treasury_data["Yield"].iloc[0]
    elif wal >= treasury_data["Maturity"].max():
        return treasury_data["Yield"].iloc[-1]
    else:
        lower = treasury_data[treasury_data["Maturity"] <= wal].iloc[-1]
        upper = treasury_data[treasury_data["Maturity"] > wal].iloc[0]
        return lower["Yield"] + (wal - lower["Maturity"]) * (upper["Yield"] - lower["Yield"]) / (upper["Maturity"] - lower["Maturity"])

bond_df["Spread (bps)"] = bond_df.apply(lambda row: (row["Yield (%)"] - interpolate_treasury_yield(row["WAL (years)"], treasury_data)) * 100, axis=1)

output_path = r"C:\Users\valim\Downloads\test\bond_spreads.xlsx"

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    combined_df.to_excel(writer, sheet_name="Treasury Curve")
    bond_df.to_excel(writer, sheet_name="Bond Data", index=False)
    bond_df[["Bond ID", "WAL (years)", "Yield (%)", "Sector", "Spread (bps)"]].to_excel(writer, sheet_name="Spreads", index=False)

    # Visualization 1: Average spread by sector
    sector_summary = bond_df.groupby("Sector").agg(
        avg_spread=("Spread (bps)", "mean"),
        avg_yield=("Yield (%)", "mean"),
        avg_wal=("WAL (years)", "mean")
    ).reset_index()

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=sector_summary, x="Sector", y="avg_spread", palette="viridis")

    for index, row in sector_summary.iterrows():
        ax.text(index, row["avg_spread"] + 0.5, f"Yield: {row['avg_yield']:.2f}%\nWAL: {row['avg_wal']:.2f} yrs", ha="center", fontsize=10)

    ax.set_title("Average Spread by Sector", fontsize=14)
    ax.set_xlabel("Sector", fontsize=12)
    ax.set_ylabel("Average Spread (bps)", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Save the plot as an image
    bar_chart_path = r"C:\Users\valim\Downloads\test\visualization.png"
    plt.savefig(bar_chart_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Append the bar chart to the Excel file
    writer.sheets["Visualizations"] = writer.book.add_worksheet("Visualizations")
    worksheet = writer.sheets["Visualizations"]
    worksheet.insert_image("A1", bar_chart_path)

 # Visualization 2: Heatmap of top opportunities
top_opportunities = bond_df.sort_values(by="Spread (bps)", ascending=False).head(10)

# Prepare annotations with simplified details
heatmap_annotations = [
    f"{row['Sector']}\nWAL: {row['WAL (years)']:.2f}\nSpread: {row['Spread (bps)']:.1f} bps"
    for _, row in top_opportunities.iterrows()
]

# Visualization 2: Heatmap of top opportunities
top_opportunities = bond_df.sort_values(by="Spread (bps)", ascending=False).head(10)

# Prepare annotations with simplified details
heatmap_annotations = [
    f"{row['Sector']}\nWAL: {row['WAL (years)']:.2f}\nSpread: {row['Spread (bps)']:.1f} bps"
    for _, row in top_opportunities.iterrows()
]

# Truncate long bond names if necessary
top_opportunities["Bond ID"] = top_opportunities["Bond ID"].apply(lambda x: x[:15] + "..." if len(x) > 15 else x)

# Create the heatmap
plt.figure(figsize=(18, 6))  # Wider figure for better spacing of bonds
sns.set_theme(style="white")

# Create a 2D array for spreads (for heatmap visualization)
spread_values = top_opportunities["Spread (bps)"].values.reshape(1, -1)

# Generate the heatmap
ax = sns.heatmap(
    spread_values,
    annot=np.array(heatmap_annotations).reshape(1, -1),
    fmt="",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Spread (bps)"},
    annot_kws={"fontsize": 10, "verticalalignment": "center", "wrap": True}  # Adjust font size and ensure wrapping
)

# Customize the heatmap
ax.set_title("Top Opportunities by Spread", fontsize=18, pad=20)
ax.set_yticks([])  # Remove y-axis labels for cleaner look
ax.set_xticklabels(
    top_opportunities["Bond ID"],  # Use bond names as x-axis labels
    rotation=45, ha="right", fontsize=11
)
ax.set_xlabel("Bonds", fontsize=14, labelpad=10)
ax.set_ylabel("")

# Adjust layout to minimize text overlap
plt.tight_layout()

# Save the heatmap as an image file
heatmap_path = r"C:\Users\valim\Downloads\test\heatmap_slide_ready.png"
plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
plt.close()

# Append the heatmap to the Excel file
with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
    # Load the workbook and sheet
    workbook = writer.book
    if "Visualizations" not in writer.sheets:
        worksheet = workbook.create_sheet("Visualizations")  # Create the sheet if it doesn't exist
    else:
        worksheet = writer.sheets["Visualizations"]

    # Insert the heatmap into the Visualizations tab
    img = Image(heatmap_path)
    worksheet.add_image(img, "A20")  # Insert image starting at cell A20

# Clean up temporary image file
os.remove(heatmap_path)

print(f"Heatmap slide-ready saved to Excel in the 'Visualizations' tab.")
