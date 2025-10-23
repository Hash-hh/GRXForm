import logging
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import time
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Molecule Design Monitor", layout="wide", page_icon="🧬")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .molecule-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


def parse_molecule_file(file_path):
    """Parse a molecule text file and extract SMILES and objectives."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract the dictionary
    try:
        mol_dict = eval(content)
        return mol_dict
    except:
        return {}


def get_epoch_files(results_path):
    """Get all epoch molecule files sorted by epoch number."""
    path = Path(results_path)
    files = list(path.glob("epoch_*_train_top_20_molecules.txt"))

    # Extract epoch numbers and sort
    epoch_files = []
    for f in files:
        match = re.search(r'epoch_(\d+)_', f.name)
        if match:
            epoch_num = int(match.group(1))
            epoch_files.append((epoch_num, f))

    epoch_files.sort(key=lambda x: x[0])
    return epoch_files


def smiles_to_image(smiles, size=(400, 400)):
    """Convert SMILES string to high-quality PIL Image."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Use higher resolution and better rendering options
        drawer = Draw.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().bondLineWidth = 2
        drawer.drawOptions().highlightBondWidthMultiplier = 20
        drawer.drawOptions().minFontSize = 15
        drawer.drawOptions().maxFontSize = 30

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Convert to PIL Image
        img_str = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_str))

        return img
    except:
        return None


def display_molecule_grid(molecules, cols=3):
    """Display molecules in a grid layout."""
    sorted_molecules = sorted(molecules.items(), key=lambda x: x[1], reverse=True)

    for i in range(0, len(sorted_molecules), cols):
        cols_list = st.columns(cols)
        for j, col in enumerate(cols_list):
            if i + j < len(sorted_molecules):
                smiles, obj = sorted_molecules[i + j]

                with col:
                    st.markdown(f"""
                    <div class="molecule-card">
                        <h4 style="color: #1f77b4;">Rank {i + j + 1}</h4>
                        <p><strong>Objective:</strong> {obj:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    img = smiles_to_image(smiles, size=(500, 500))  # Higher resolution
                    if img:
                        st.image(img, width='stretch')  # Fixed warning
                    else:
                        st.error("Failed to render molecule")

                    with st.expander("Show SMILES"):
                        st.code(smiles, language=None)


def create_objective_plot(epochs_data, selected_epoch=None):
    """Create an interactive plot of objective values over epochs."""
    fig = go.Figure()

    epochs = []
    best_objs = []
    mean_objs = []
    worst_objs = []
    top_molecules = []

    for epoch, molecules in epochs_data.items():
        if molecules:
            objs = list(molecules.values())
            epochs.append(epoch)
            best_objs.append(max(objs))
            mean_objs.append(np.mean(objs))
            worst_objs.append(min(objs))
            top_smiles = max(molecules.items(), key=lambda x: x[1])[0]
            top_molecules.append(f"Epoch {epoch}: {top_smiles[:50]}...")

    # Add traces
    fig.add_trace(go.Scatter(
        x=epochs, y=best_objs,
        mode='lines+markers',
        name='Best Objective',
        line=dict(color='green', width=3),
        marker=dict(size=10),
        customdata=[[top_molecules[i]] for i in range(len(epochs))],
        hovertemplate='<b>Epoch %{x}</b><br>Best: %{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=epochs, y=mean_objs,
        mode='lines+markers',
        name='Mean Objective',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        hovertemplate='<b>Epoch %{x}</b><br>Mean: %{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=epochs, y=worst_objs,
        mode='lines+markers',
        name='Worst Objective',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        hovertemplate='<b>Epoch %{x}</b><br>Worst: %{y:.4f}<extra></extra>'
    ))

    if selected_epoch is not None and selected_epoch in epochs:
        idx = epochs.index(selected_epoch)
        fig.add_trace(go.Scatter(
            x=[selected_epoch],
            y=[best_objs[idx]],
            mode='markers',
            name='Selected Epoch',
            marker=dict(size=20, color='orange', symbol='star'),
            showlegend=False
        ))

    fig.update_layout(
        title='Objective Values Over Training Epochs',
        xaxis_title='Epoch',
        yaxis_title='Objective Value',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )

    return fig


def create_distribution_plot(molecules):
    """Create a histogram of objective values."""
    objs = list(molecules.values())

    fig = go.Figure(data=[go.Histogram(
        x=objs,
        nbinsx=20,
        marker_color='lightblue',
        marker_line_color='darkblue',
        marker_line_width=1.5
    )])

    fig.update_layout(
        title='Distribution of Objective Values',
        xaxis_title='Objective Value',
        yaxis_title='Count',
        height=300,
        template='plotly_white'
    )

    return fig



def main():
    st.markdown('<h1 class="main-header">🧬 Molecule Design Monitor</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Get default path relative to project root
    default_path = str(Path(__file__).parent.parent / "results" / "2025-10-19--22-08-48")

    results_path = st.sidebar.text_input(
        "Results Folder Path",
        value=default_path,
        help="Path to the folder containing epoch result files"
    )

    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=10,
        disabled=not auto_refresh
    )

    cols_per_row = st.sidebar.slider("Molecules per row", 2, 5, 3)

    # Check if path exists
    if not Path(results_path).exists():
        st.error(f"❌ Path '{results_path}' does not exist!")
        return

    # Get epoch files
    epoch_files = get_epoch_files(results_path)

    if not epoch_files:
        st.warning("⚠️ No epoch files found in the specified directory.")
        return

    # Load all epochs data
    epochs_data = {}
    for epoch_num, file_path in epoch_files:
        molecules = parse_molecule_file(file_path)
        epochs_data[epoch_num] = molecules

    # Main metrics
    latest_epoch = max(epochs_data.keys())
    total_epochs = len(epochs_data)
    latest_molecules = epochs_data[latest_epoch]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Latest Epoch", latest_epoch)
    with col2:
        st.metric("Total Epochs", total_epochs)
    with col3:
        if latest_molecules:
            best_obj = max(latest_molecules.values())
            st.metric("Best Objective (Latest)", f"{best_obj:.4f}")
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

    # Progress tracking
    st.markdown("---")
    st.subheader("📊 Training Progress")

    # Create objective plot
    selected_epoch = st.select_slider(
        "Select Epoch to View",
        options=sorted(epochs_data.keys()),
        value=latest_epoch
    )

    fig_progress = create_objective_plot(epochs_data, selected_epoch)
    st.plotly_chart(fig_progress, use_container_width=True)

    # Display selected epoch details
    st.markdown("---")
    st.subheader(f"🔬 Epoch {selected_epoch} - Top 20 Molecules")

    selected_molecules = epochs_data[selected_epoch]

    if selected_molecules:
        # Statistics
        col1, col2 = st.columns(2)

        with col1:
            objs = list(selected_molecules.values())
            stats_df = pd.DataFrame({
                'Metric': ['Best', 'Mean', 'Worst', 'Std Dev'],
                'Value': [
                    f"{max(objs):.4f}",
                    f"{np.mean(objs):.4f}",
                    f"{min(objs):.4f}",
                    f"{np.std(objs):.4f}"
                ]
            })
            st.dataframe(stats_df, width='stretch', hide_index=True)

        with col2:
            fig_dist = create_distribution_plot(selected_molecules)
            st.plotly_chart(fig_dist, use_container_width=True)

        # Molecule grid
        st.markdown("### Molecule Structures")
        display_molecule_grid(selected_molecules, cols=cols_per_row)

        # Download data
        st.markdown("---")
        st.subheader("💾 Export Data")

        df = pd.DataFrame([
            {'Rank': i + 1, 'SMILES': smiles, 'Objective': obj}
            for i, (smiles, obj) in enumerate(
                sorted(selected_molecules.items(), key=lambda x: x[1], reverse=True)
            )
        ])

        csv = df.to_csv(index=False)
        st.download_button(
            label=f"Download Epoch {selected_epoch} Data (CSV)",
            data=csv,
            file_name=f"epoch_{selected_epoch}_molecules.csv",
            mime="text/csv"
        )

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
