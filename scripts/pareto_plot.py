"""
Plot for OCR performance vs cost Pareto frontier figure for NeurIPS paper.

Invocation:
    python ocr_pareto_frontier.py output/
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib.ticker as ticker

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("output_dir", type=str, help="Path to the output directory")
ap.add_argument(
    "--font-path",
    type=str,
    help="Path to the font file",
    default=None,
)
args = ap.parse_args()

# Add custom font if provided
if args.font_path:
    font_manager.fontManager.addfont(args.font_path)
    plt.rcParams["font.family"] = "Manrope"
    plt.rcParams["font.weight"] = "medium"

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)
OUTPUT_PATHS = [f"{args.output_dir}/ocr_pareto.pdf", f"{args.output_dir}/ocr_pareto.png"]

# Define column names
MODEL_COLUMN_NAME = "Model"
CATEGORY_COLUMN_NAME = "Category"
COST_COLUMN_NAME = "Cost_Per_Million"
PERF_COLUMN_NAME = "Performance"
COLOR_COLUMN_NAME = "Color"
OFFSET_COLUMN_NAME = "Label_Offset"
MARKER_COLUMN_NAME = "Marker"

# Define colors
DARK_BLUE = "#093235"
DARK_GREEN = "#255457"
LIGHT_GREEN = "#6FE0BA"
LIGHT_PINK = "#F697C4"
DARK_PINK = "#F0529C"
YELLOW = "#fff500"
ORANGE = "#f65834"
DARK_TEAL = "#0a3235"
OFF_WHITE = "#faf2e9"
TEAL = "#105257"
PURPLE = "#b11be8"
GREEN = "#0fcb8c"

# Create dataframe with OCR model data
data = {
    MODEL_COLUMN_NAME: [
        "GPT-4o",
        "GPT-4o Batch",
        "Mistral OCR",
        "MinerU",
        "Gemini Flash 2",
        "Gemini Flash 2 Batch",
        "marker v1.6.2",
        "Ours (A100)",
        "Ours (L40S)",
        "Ours (H100)",
        "Qwen2VL",
        "Qwen2.5VL"
    ],
    COST_COLUMN_NAME: [
        12480,
        6240,
        1000,
        596,
        499,
        249,
        235,
        270,
        190,
        190,
        190,  # Same cost as Ours (L40S)
        190   # Same cost as Ours (L40S)
    ],
    PERF_COLUMN_NAME: [
        69.9,  # GPT-4o (Anchored)
        69.9,  # Same performance for batch
        72.0,  # Mistral OCR API
        61.5,  # MinerU
        63.8,  # Gemini Flash 2 (Anchored)
        63.8,  # Same performance for batch
        59.4,  # marker v1.6.2
        77.4,  # Ours (performance is the same across hardware)
        77.4,  # Ours (performance is the same across hardware)
        77.4,  # Ours (performance is the same across hardware)
        31.5,  # Qwen2VL
        65.5   # Qwen2.5VL
    ]
}

df = pd.DataFrame(data)

# Add category information
model_categories = {
    "GPT-4o": "Commercial API",
    "GPT-4o Batch": "Commercial API",
    "Mistral OCR": "Commercial API",
    "MinerU": "Open Source",
    "Gemini Flash 2": "Commercial API",
    "Gemini Flash 2 Batch": "Commercial API",
    "marker v1.6.2": "Open Source",
    "Ours (A100)": "Ours",
    "Ours (L40S)": "Ours",
    "Ours (H100)": "Ours",
    "Qwen2VL": "Open Source", 
    "Qwen2.5VL": "Open Source"
}

df[CATEGORY_COLUMN_NAME] = df[MODEL_COLUMN_NAME].map(model_categories)

# Category colors
category_colors = {
    "Commercial API": DARK_BLUE,
    "Open Source": LIGHT_GREEN,
    "Ours": DARK_PINK
}

df[COLOR_COLUMN_NAME] = df[CATEGORY_COLUMN_NAME].map(category_colors)

# Define marker types
category_markers = {
    "Commercial API": "o",
    "Open Source": "s",
    "Ours": "*"
}

df[MARKER_COLUMN_NAME] = df[CATEGORY_COLUMN_NAME].map(category_markers)

# Define marker sizes
category_marker_sizes = {
    "Commercial API": 60,
    "Open Source": 70,
    "Ours": 150
}

# Define text colors
category_text_colors = {
    "Commercial API": DARK_TEAL,
    "Open Source": DARK_TEAL,
    "Ours": "#a51c5c"  # darker pink
}

# Label offsets for better readability
model_label_offsets = {
    "GPT-4o": [10, 5],
    "GPT-4o Batch": [10, 5],
    "Mistral OCR": [-20, 10],
    "MinerU": [-40, 5],
    "Gemini Flash 2": [10, -10],
    "Gemini Flash 2 Batch": [10, 0],
    "marker v1.6.2": [-50, -10],
    "Ours (A100)": [-20, 10],
    "Ours (L40S)": [10, 5],
    "Ours (H100)": [-60, -5],
    "Qwen2VL": [-50, 5],
    "Qwen2.5VL": [10, -10]
}

df[OFFSET_COLUMN_NAME] = df[MODEL_COLUMN_NAME].map(model_label_offsets)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each category
categories = df[CATEGORY_COLUMN_NAME].unique()
for category in categories:
    mask = df[CATEGORY_COLUMN_NAME] == category
    data = df[mask]
    plt.scatter(
        data[COST_COLUMN_NAME],
        data[PERF_COLUMN_NAME],
        label=category,
        c=data[COLOR_COLUMN_NAME],
        marker=category_markers[category],
        alpha=1.0,
        s=category_marker_sizes[category],
    )

# Add labels for each point
FONTSIZE = 9
for idx, row in df.iterrows():
    plt.annotate(
        row[MODEL_COLUMN_NAME],
        (row[COST_COLUMN_NAME], row[PERF_COLUMN_NAME]),
        xytext=row[OFFSET_COLUMN_NAME],
        textcoords="offset points",
        fontsize=FONTSIZE,
        alpha=1.0,
        weight="medium",
        color=category_text_colors[row[CATEGORY_COLUMN_NAME]],
    )

# Set up axes
plt.xscale('log')  # Use log scale for cost
plt.ylim(30, 80)   # Set y-axis limits from 30 to 80 to include Qwen2VL
plt.grid(True, which="both", ls=":", color=TEAL, alpha=0.2)

# Format y-axis to show percentages without scientific notation
def percent_formatter(y, pos):
    return f'{y:.1f}%'

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(percent_formatter))

# Format x-axis to show dollar amounts
def dollar_formatter(x, pos):
    return f'${x:,.0f}'

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))

# Add labels and title
plt.xlabel("Cost per Million Pages (USD, log scale)", fontsize=10, weight="medium")
plt.ylabel("Overall Performance (Pass Rate %)", fontsize=10, weight="medium")
plt.title("OCR Engines: Performance vs. Cost", fontsize=12, weight="medium")

# Remove spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Create Pareto frontier
# Sort by cost, ascending
frontier_models = []
pareto_df = df.copy()
pareto_df = pareto_df.sort_values(by=COST_COLUMN_NAME)

# Find Pareto optimal points
max_perf = 0
for idx, row in pareto_df.iterrows():
    if row[PERF_COLUMN_NAME] > max_perf:
        max_perf = row[PERF_COLUMN_NAME]
        frontier_models.append(row[MODEL_COLUMN_NAME])

# Get the frontier points
frontier_df = df[df[MODEL_COLUMN_NAME].isin(frontier_models)].sort_values(by=COST_COLUMN_NAME)

# Create and add the Pareto frontier polygon
xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()

# Create polygon vertices for the Pareto frontier
# Sort frontier_df by cost for correct polygon creation
frontier_df = frontier_df.sort_values(by=COST_COLUMN_NAME)

# Start with the points from the Pareto frontier
X = []
for _, row in frontier_df.iterrows():
    X.append([row[COST_COLUMN_NAME], row[PERF_COLUMN_NAME]])

# Convert to numpy array
X = np.array(X)

# Add points to close the polygon at the bottom
bottom_y = 30  # Minimum y-value for the plot
X = np.vstack([
    X,                                 # Pareto optimal points
    [X[-1, 0], bottom_y],              # Bottom right corner
    [X[0, 0], bottom_y]                # Bottom left corner
])

# # Add the polygon
# polygon = plt.Polygon(
#     X, facecolor=YELLOW, alpha=0.15, zorder=-1, edgecolor=ORANGE, linestyle="--", linewidth=1.5
# )
# plt.gca().add_patch(polygon)

# Add the legend with custom ordering
handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ["Ours", "Open Source", "Commercial API"]
label_to_handle = dict(zip(labels, handles))
ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
ordered_labels = [label for label in desired_order if label in labels]

plt.legend(
    ordered_handles,
    ordered_labels,
    loc="upper right",
    fontsize=10,
    frameon=True,
    framealpha=0.9,
    edgecolor=TEAL,
    facecolor="white"
)

# Adjust layout
plt.tight_layout()

# Save the figure
for output_path in OUTPUT_PATHS:
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Plot saved to {', '.join(OUTPUT_PATHS)}")