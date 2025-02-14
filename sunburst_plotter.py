import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class SunburstPlotter:
    def __init__(self, drugs_H, targets_W, X, Xr, Xc, gamma, df_predictions):
        """Initialize the object with precomputed matrices and data."""
        self.drugs_H = drugs_H
        self.targets_W = targets_W
        self.X = X
        self.Xr = Xr
        self.Xc = Xc
        self.gamma = gamma
        self.df_predictions = df_predictions

        # Compute sum of all betas and alphas
        self.sum_betas = sum(float(value) for key, value in drugs_H.items() if "beta" in key)
        self.sum_alphas = sum(float(value) for key, value in targets_W.items() if "alpha" in key)

        # Compute H and W matrices
        self.H = self._compute_H()
        self.W = self._compute_W()

    def _compute_H(self):
        """Compute H using the correct weighted sum formula."""
        H = np.zeros((self.X.shape[0], self.X.shape[0]), dtype=np.float64)
        for key, matrix in self.drugs_H.items():
            if "beta" in key:
                H += (matrix.astype(np.float64) / self.sum_betas) * self.drugs_H[key.replace("beta_", "")].astype(np.float64)
        return H

    def _compute_W(self):
        """Compute W using the correct weighted sum formula."""
        W = np.zeros((self.X.shape[1], self.X.shape[1]), dtype=np.float64)
        for key, matrix in self.targets_W.items():
            if "alpha" in key:
                W += (matrix.astype(np.float64) / self.sum_alphas) * self.targets_W[key.replace("alpha_", "")].astype(np.float64)
        return W

    def generate_sunburst_plot(self, drug_name, protein_name):
        """Generates and saves a sunburst plot for a given drug and protein."""
        
        # Get indices for the drug and target
        drug_i = self.df_predictions[(self.df_predictions["drug_name"] == drug_name) & 
                                     (self.df_predictions["protein_name"] == protein_name)]["Drug_Index"].values[0]
        target_j = self.df_predictions[(self.df_predictions["drug_name"] == drug_name) & 
                                       (self.df_predictions["protein_name"] == protein_name)]["Target_Index"].values[0]

        # Find interacting drugs and targets
        interacting_drugs = np.where(self.X[:, target_j] > 0)[0]
        interacting_targets = np.where(self.X[drug_i, :] > 0)[0]

        # Compute contributions
        H_contributions = {}
        for key in self.drugs_H.keys():
            if "beta" in key:
                original_key = key.replace("beta_", "")
                for drug in interacting_drugs:
                    sim_drug_name = self.df_predictions[self.df_predictions["Drug_Index"] == drug]["drug_name"].values[0]
                    H_contributions[(sim_drug_name, original_key)] = (
                        (self.drugs_H[key] / self.sum_betas) * self.drugs_H[original_key][drug_i, drug] * self.Xr[drug, target_j]
                    )

        W_contributions = {}
        for key in self.targets_W.keys():
            if "alpha" in key:
                original_key = key.replace("alpha_", "")
                for target in interacting_targets:
                    sim_target_name = self.df_predictions[self.df_predictions["Target_Index"] == target]["protein_name"].values[0]
                    W_contributions[(sim_target_name, original_key)] = (
                        (self.targets_W[key] / self.sum_alphas) * self.targets_W[original_key][target_j, target] * self.Xc[drug_i, target]
                    )

        # Create the dataframe for the sunburst plot
        data = []
        for (sim_drug_name, similarity_type), value in H_contributions.items():
            data.append([similarity_type, (1 - self.gamma) * value, sim_drug_name, drug_name])

        for (sim_target_name, similarity_type), value in W_contributions.items():
            data.append([similarity_type, self.gamma * value, sim_target_name, protein_name])

        df_sunburst = pd.DataFrame(data, columns=["similarity", "value", "similar_entity", "side"])

        # Define the color mapping
        color_map = {
            "drug_name": "#5ab4ac",  
            "target_name": "#e9a3c9",  
            "similar_drugs": "#d8daeb",  
            "similar_targets": "#fee08b",  
            "drug_covariance": "#d9f0d3",  
            "chemical": "#fdb863",  
            "side_effect": "#abd9e9",  
            "ddi": "#f4a582",  
            "target_covariance": "#e66101",  
            "sequence": "#0571b0",  
            "ppi": "#008837",  
        }

        # Ensure all unique values in 'side' and 'similar_entity' have assigned colors
        unique_sides = df_sunburst['side'].unique()
        unique_entities = df_sunburst['similar_entity'].unique()

        for side in unique_sides:
            if side not in color_map:
                color_map[side] = color_map['drug_name'] if drug_name == side else color_map['target_name']

        for entity in unique_entities:
            if entity not in color_map:
                color_map[entity] = color_map['similar_drugs'] if df_sunburst[df_sunburst.similar_entity == entity].side.values[0] == drug_name else color_map['similar_targets']

        # Create the sunburst plot
        fig = px.sunburst(
            df_sunburst,
            path=['side', 'similar_entity', 'similarity'],
            values='value',
            color=df_sunburst['side'],
            color_discrete_map=color_map
        )

        # Force update of colors for all levels
        fig.update_traces(
            marker=dict(colors=[color_map.get(cat, "#cccccc") for cat in fig.data[-1].labels]),
            textinfo="label+percent parent"  # Show labels only for inner rings
            )

        # Create legend manually for similarity types (outer layer)
        legend_traces = []
        for similarity, color in color_map.items():
            if similarity in df_sunburst['similarity'].unique():
                legend_traces.append(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=15, color=color),
                    name=similarity
                ))

        # Add custom legend traces
        fig.add_traces(legend_traces)

        # Update layout
        fig.update_layout(
            showlegend=True,
            legend_title_text="Similarity Type",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white"
        )

        # Save and return file path
        filename = f"{drug_name}_{protein_name}_sunburst_plot.html"
        fig.write_html(filename)
        return filename
