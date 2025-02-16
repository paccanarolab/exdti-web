from fastapi import FastAPI  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from fastapi import BackgroundTasks  # type: ignore
from fastapi import Request  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import json
import pickle as pkl
import zipfile
from jinja2 import Template
from sunburst_plotter import SunburstPlotter
import os

# Define file paths
zip_path = "data_for_plotting.zip"
extract_path = "data_for_plotting.pkl"


# Unzip the file if it doesn't exist yet
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")  # Extracts in the same directory

# Load the extracted pickle file
with open(extract_path, "rb") as f:
    data = pkl.load(f)

drugs_H = data["drugs_H"]
targets_W = data["targets_W"]
X = data["X"]
Xr = data["Xr"]
Xc = data["Xc"]
df_predictions = data["df_predictions"]
gamma = data["gamma"]
# replace in df_predictions all / with _
df_predictions.replace("/", "_", regex=True, inplace=True)
app = FastAPI()

plotter = SunburstPlotter(drugs_H, targets_W, X, Xr, Xc, gamma, df_predictions)

# Serve static files
app.mount("/exdtiweb/static", StaticFiles(directory="/app/static"), name="static")

# Set up template rendering
templates = Jinja2Templates(directory="templates")

# async function generatePlot() {
#             if (!selectedDrug || !selectedTarget) return alert("Select both a drug and a target.");

#             let plotUrl = `/plot/${selectedDrug}/${selectedTarget}`;
#             window.open(plotUrl, "_blank");  // Opens in a new tab
#         }


@app.get("/", response_class=FileResponse)
def serve_homepage(request: Request):
    """Serve the HTML page from templates."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "static_url": "/exdtiweb/static"}
    )


@app.get("/drugs")
def get_drugs():
    """Return a list of unique drug names in alphabetical order, excluding None values."""
    drugs = df_predictions["drug_name"].dropna().unique().tolist()
    return {"drugs": sorted(drugs)}


@app.get("/targets")
def get_targets():
    """Return a list of unique target (protein) names in alphabetical order, excluding None values."""
    targets = df_predictions["protein_name"].dropna().unique().tolist()
    return {"targets": sorted(targets)}


@app.get("/targets/{drug_name}")
def get_targets_for_drug(drug_name: str):
    """Return ranked targets for a given drug."""
    ranked_targets = df_predictions[
        df_predictions["drug_name"] == drug_name
    ].sort_values("Predicted_Score", ascending=False)[
        ["protein_name", "Predicted_Score"]
    ]
    return {"targets": ranked_targets.to_dict(orient="records")}


@app.get("/drugs/{protein_name}")
def get_drugs_for_target(protein_name: str):
    """Return ranked drugs for a given target."""
    ranked_drugs = df_predictions[
        df_predictions["protein_name"] == protein_name
    ].sort_values("Predicted_Score", ascending=False)[["drug_name", "Predicted_Score"]]
    return {"drugs": ranked_drugs.to_dict(orient="records")}


@app.get("/prediction_rank/{drug_name}/{protein_name}")
def get_prediction_rank(drug_name: str, protein_name: str):
    """Compute percentile & ranking of a drug-target prediction."""

    # Get the predicted score for the selected pair
    selected_row = df_predictions[
        (df_predictions["drug_name"] == drug_name)
        & (df_predictions["protein_name"] == protein_name)
    ]

    if selected_row.empty:
        return {"error": "Prediction not found"}

    selected_score = selected_row["Predicted_Score"].values[0]

    # Compute global percentile
    all_scores = df_predictions["Predicted_Score"]
    global_percentile = round((all_scores < selected_score).mean() * 100, 5)

    # Compute ranking among targets for this drug
    drug_subset = df_predictions[df_predictions["drug_name"] == drug_name].sort_values(
        "Predicted_Score", ascending=False
    )
    drug_rank = (drug_subset["Predicted_Score"] >= selected_score).sum()
    total_targets = len(drug_subset)

    # Compute ranking among drugs for this target
    target_subset = df_predictions[
        df_predictions["protein_name"] == protein_name
    ].sort_values("Predicted_Score", ascending=False)
    target_rank = (target_subset["Predicted_Score"] >= selected_score).sum()
    total_drugs = len(target_subset)

    return {
        "global_percentile": f"{global_percentile:.2f}th percentile",
        "drug_target_rank": f"{drug_rank}/{total_targets}",
        "target_drug_rank": f"{target_rank}/{total_drugs}",
    }


@app.get("/known_interactions/{entity_type}/{entity_name}")
def get_known_interactions(entity_type: str, entity_name: str):
    """Fetch known interactions from X matrix based on drug or target selection."""

    if entity_type == "drug":
        # Get drug index
        drug_row = df_predictions[df_predictions["drug_name"] == entity_name]
        if drug_row.empty:
            return {"interactions": []}

        drug_idx = drug_row["Drug_Index"].values[0]

        # Find known targets (nonzero values in X row)
        known_targets = np.where(X[drug_idx, :] > 0)[0]

        # Convert indices to target names
        known_target_names = (
            df_predictions[df_predictions["Target_Index"].isin(known_targets)][
                "protein_name"
            ]
            .unique()
            .tolist()
        )
        return {"interactions": known_target_names}

    elif entity_type == "target":
        # Get target index
        target_row = df_predictions[df_predictions["protein_name"] == entity_name]
        if target_row.empty:
            return {"interactions": []}

        target_idx = target_row["Target_Index"].values[0]

        # Find known drugs (nonzero values in X column)
        known_drugs = np.where(X[:, target_idx] > 0)[0]

        # Convert indices to drug names
        known_drug_names = (
            df_predictions[df_predictions["Drug_Index"].isin(known_drugs)]["drug_name"]
            .unique()
            .tolist()
        )
        return {"interactions": known_drug_names}

    return {"interactions": []}


@app.get("/plot/{drug_name}/{protein_name}")
def generate_plot(drug_name: str, protein_name: str, background_tasks: BackgroundTasks):
    """Generate sunburst plot, serve it, and delete it after display."""

    plot_file = plotter.generate_sunburst_plot(drug_name, protein_name)

    # Schedule file deletion after response is sent
    background_tasks.add_task(os.remove, plot_file)

    return FileResponse(plot_file)
