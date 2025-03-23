# -*- coding: utf-8 -*-

"""
This script visualizes RRCS data for specific residues, combining a bar plot with error bars,
a scatter plot of individual values, and a density plot of RRCS changes over time by residue
properties. It processes data from an HDF5 file and a PDB file, with optional residue pair
specification via a text file.
"""

import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import MDAnalysis as mda

# Define amino acid properties and colors
AMINO_ACID_PROPERTIES = {
    'ALA': 'Hydrophobic', 'VAL': 'Hydrophobic', 'LEU': 'Hydrophobic', 'ILE': 'Hydrophobic',
    'PRO': 'Special', 'MET': 'Hydrophobic',
    'PHE': 'Aromatic', 'TYR': 'Aromatic', 'TRP': 'Aromatic',
    'HIS': 'Charged', 'LYS': 'Charged', 'ARG': 'Charged',
    'ASP': 'Charged', 'GLU': 'Charged',
    'ASN': 'Polar uncharged', 'GLN': 'Polar uncharged', 'SER': 'Polar uncharged', 'THR': 'Polar uncharged',
    'CYS': 'Special', 'GLY': 'Special', 'SEC': 'Special', 'UNK': 'Special',
    'PL1': 'Hydrophobic', 'AIB': 'Hydrophobic', 'ALY': 'Special'
}

PROPERTY_COLORS = {
    'Hydrophobic': '#7A7A7A',  # Gray
    'Aromatic': '#800080',     # Purple
    'Charged': '#FF0000',      # Red
    'Polar uncharged': '#008000',  # Green
    'Special': '#FFA500'       # Orange
}

THREE_TO_ONE_LETTER = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'AIB': 'Aib', 'PL1': 'PL1', 'ALY': 'ALY'
}

def get_amino_acid_property(resname):
    """Get the property of an amino acid based on its name.

    Args:
        resname (str): Three-letter amino acid code.

    Returns:
        str: Property of the amino acid (e.g., 'Hydrophobic').
    """
    return AMINO_ACID_PROPERTIES.get(resname.upper(), 'Special')

def modify_label(label):
    """Modify residue labels to a concise format (e.g., 'A123' for '123_ALA').

    Args:
        label (str): Original residue label in format 'resid_resname'.

    Returns:
        str: Modified label using one-letter code and residue ID.
    """
    parts = label.split('_')
    if len(parts) == 2:
        resid, resname = parts
        one_letter = THREE_TO_ONE_LETTER.get(resname.upper(), resname)
        return f"{one_letter}{resid}"
    return label

def parse_arguments():
    """Parse command-line arguments.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize RRCS data with bar, scatter, and density plots.")
    parser.add_argument('--h5_file', type=str, required=True, help="Path to the HDF5 file containing RRCS data.")
    parser.add_argument('--pdb_file', type=str, required=True, help="Path to the PDB file for residue information.")
    parser.add_argument('--res_file', type=str, help="Path to the file containing residue pair indices.")
    parser.add_argument('--bt', type=float, default=500000.0, help="Start time in ps (default: 500000.0).")
    parser.add_argument('--et', type=float, default=1000000.0, help="End time in ps (default: 1000000.0).")
    parser.add_argument('--output_dir', type=str, default='.', help="Directory to save output plot (default: current directory).")
    parser.add_argument('--frames_per_average', type=int, default=100, help="Number of frames to average over (default: 100).")
    parser.add_argument('--window_size', type=int, default=1000, help="Window size for smoothing density data (default: 1000).")
    parser.add_argument('--mean_threshold', type=float, default=0.8, help="Mean RRCS threshold for filtering residues (default: 0.8).")
    parser.add_argument('--resid_threshold', type=int, default=137, help="Residue ID threshold for filtering (default: 137).")
    parser.add_argument('--gap_fraction', type=float, default=0.5, help="Fraction of residue duration for gaps (default: 0.5).")
    parser.add_argument('--jitter_strength', type=float, default=0.25, help="Strength of jitter for scatter plot (default: 0.25).")

    if len(sys.argv) == 1:
        print("Error: No arguments provided. Displaying help:\n")
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def read_residue_pairs(res_file):
    """Read and parse the residue pair indices file.

    Args:
        res_file (str): Path to the residue pair indices file.

    Returns:
        set: A set of tuples representing residue pairs, or None if file not provided/exists.
    """
    if not res_file or not os.path.exists(res_file):
        print(f"Warning: Residue pair file {res_file} not provided or does not exist. All residue pairs will be considered.")
        return None

    res_pairs = set()
    with open(res_file, 'r') as f:
        for line in f:
            line = line.split(';')[0].strip()  # Remove comments
            if not line:
                continue
            if '$' in line:
                parts = line.split('$')
                if len(parts) != 2:
                    print(f"Error: Invalid line in residue pair file: {line}")
                    continue
                res1 = parse_residue_list(parts[0].strip())
                res2 = parse_residue_list(parts[1].strip())
                for r1 in res1:
                    for r2 in res2:
                        res_pairs.add((r1, r2))
            else:
                res_list = parse_residue_list(line)
                for i in range(len(res_list)):
                    for j in range(i + 1, len(res_list)):
                        res_pairs.add((res_list[i], res_list[j]))
    return res_pairs

def parse_residue_list(res_str):
    """Parse a string of residue IDs, supporting ranges.

    Args:
        res_str (str): String containing residue IDs or ranges (e.g., '1-5 7').

    Returns:
        list: List of integer residue IDs.
    """
    res_list = []
    for part in res_str.split():
        if '-' in part:
            start, end = map(int, part.split('-'))
            res_list.extend(range(start, end + 1))
        else:
            res_list.append(int(part))
    return res_list

def main():
    """Main function to process RRCS data and generate combined visualization."""
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load HDF5 file
    try:
        with h5py.File(args.h5_file, 'r') as hf:
            residue_pairs_A = hf['residue_pairs_A'][:]
            residue_pairs_B = hf['residue_pairs_B'][:]
            times = hf['times'][:]
            rrcs = hf['rrcs'][:]
    except FileNotFoundError:
        print(f"Error: HDF5 file not found: {args.h5_file}")
        sys.exit(1)

    # Load PDB file
    try:
        u_pdb = mda.Universe(args.pdb_file)
        proteinA_pdb = u_pdb.select_atoms("(protein or resname PL1 or resname AIB or resname ALY) and chainid A")
        proteinB_pdb = u_pdb.select_atoms("protein and chainid B")
    except FileNotFoundError:
        print(f"Error: PDB file not found: {args.pdb_file}")
        sys.exit(1)

    # Map residues to properties
    resid_to_resname_A = {res.resid: res.resname for res in proteinA_pdb.residues}
    resid_to_resname_B = {res.resid: res.resname for res in proteinB_pdb.residues}
    resid_to_property_A = {resid: get_amino_acid_property(resname) for resid, resname in resid_to_resname_A.items()}
    resid_to_property_B = {resid: get_amino_acid_property(resname) for resid, resname in resid_to_resname_B.items()}

    # Filter data based on time range
    time_indices = np.where((times >= args.bt) & (times <= args.et))[0]
    filtered_times = times[time_indices]
    filtered_rrcs = rrcs[:, time_indices]

    # Read residue pairs if provided
    res_pairs = read_residue_pairs(args.res_file)
    if res_pairs is None:
        unique_resid_B = np.unique(residue_pairs_B)
    else:
        unique_resid_B = sorted(set(pair[1] for pair in res_pairs if pair[0] in resid_to_resname_A))

    # Process RRCS data for chain B
    rrcs_sum_B_filtered = {resid: np.zeros(len(filtered_times), dtype=np.float32) for resid in unique_resid_B}
    for i in tqdm(range(len(residue_pairs_B)), desc="Accumulating RRCS sums"):
        b_resid = residue_pairs_B[i]
        if b_resid in unique_resid_B:
            rrcs_sum_B_filtered[b_resid] += filtered_rrcs[i]

    # Average RRCS sums
    averaged_rrcs_sum_B = {}
    averaged_rrcs_values_B = {}
    for resid in unique_resid_B:
        rrcs_sum = rrcs_sum_B_filtered[resid]
        num_complete_averages = len(rrcs_sum) // args.frames_per_average
        if num_complete_averages == 0:
            print(f"Warning: Residue {resid} has less than {args.frames_per_average} frames. Skipping.")
            continue
        rrcs_sum_trimmed = rrcs_sum[:num_complete_averages * args.frames_per_average]
        averaged_rrcs = rrcs_sum_trimmed.reshape(-1, args.frames_per_average).mean(axis=1)
        averaged_rrcs_sum_B[resid] = averaged_rrcs.mean()
        averaged_rrcs_values_B[resid] = averaged_rrcs

    # Prepare data for plotting
    data_for_plot_B = []
    for resid in unique_resid_B:
        if resid not in averaged_rrcs_sum_B:
            continue
        mean = averaged_rrcs_sum_B[resid]
        se = np.std(averaged_rrcs_values_B[resid], ddof=1) / np.sqrt(len(averaged_rrcs_values_B[resid]))
        resname = resid_to_resname_B.get(resid, f"Res{resid}")
        property_B = get_amino_acid_property(resname)
        label = f"{resid}_{resname}"
        data_for_plot_B.append({'Resid': resid, 'Residue': label, 'Mean': mean, 'SE': se, 'Property': property_B, 'Values': averaged_rrcs_values_B[resid]})

    # Create DataFrames
    df1_B = pd.DataFrame({
        'Residue': [item['Residue'] for item in data_for_plot_B],
        'Mean': [item['Mean'] for item in data_for_plot_B],
        'SE': [item['SE'] for item in data_for_plot_B],
        'Property': [item['Property'] for item in data_for_plot_B],
        'Resid': [item['Resid'] for item in data_for_plot_B]
    })
    df2_B = pd.DataFrame({
        'Residue': np.repeat([item['Residue'] for item in data_for_plot_B], len(data_for_plot_B[0]['Values'])),
        'Value': np.concatenate([item['Values'] for item in data_for_plot_B]),
        'Property': np.repeat([item['Property'] for item in data_for_plot_B], len(data_for_plot_B[0]['Values']))
    })

    # Assign colors
    df1_B['Color'] = df1_B['Property'].map(PROPERTY_COLORS).fillna('#000000')
    df2_B['Color'] = df2_B['Property'].map(PROPERTY_COLORS).fillna('#000000')

    # Filter residues
    df1_filtered_B_temp = df1_B[df1_B['Mean'] > args.mean_threshold].reset_index(drop=True)
    df1_filtered_B = df1_filtered_B_temp[df1_filtered_B_temp['Resid'] > args.resid_threshold].reset_index(drop=True)
    filtered_residues_B = df1_filtered_B['Residue'].tolist()
    df2_filtered_B = df2_B[df2_B['Residue'].isin(filtered_residues_B)].copy()

    # Modify labels
    df1_filtered_B['Modified_Residue'] = df1_filtered_B['Residue'].apply(modify_label)
    df2_filtered_B['Modified_Residue'] = df2_filtered_B['Residue'].apply(modify_label)

    # Prepare data for density plot
    properties = list(PROPERTY_COLORS.keys())
    rrcs_sum_B_properties = {resid: {prop: np.zeros(len(filtered_times), dtype=np.float32) for prop in properties}
                             for resid in df1_filtered_B["Resid"]}
    for i in tqdm(range(len(residue_pairs_B)), desc="Accumulating RRCS sums by property"):
        b_resid = residue_pairs_B[i]
        if b_resid not in df1_filtered_B["Resid"].tolist():
            continue
        a_resid = residue_pairs_A[i]
        property_A = resid_to_property_A.get(a_resid, 'Special')
        rrcs_sum_B_properties[b_resid][property_A] += filtered_rrcs[i]

    # Calculate time parameters for density plot
    simulation_duration = (args.et - args.bt) / 1000.0  # in ns
    residue_duration = simulation_duration
    gap_duration = args.gap_fraction * residue_duration
    time_per_residue_with_gap = residue_duration + gap_duration
    num_residues = len(df1_filtered_B["Resid"])
    sorted_resid_B = sorted(rrcs_sum_B_properties.keys())

    # Initialize density data
    smoothed_density_data_per_property = {prop: [] for prop in properties}
    x_list = []
    concatenated_labels = []

    # Process each residue for density plot
    for residue_index, resid in enumerate(sorted_resid_B):
        resname = resid_to_resname_B.get(resid, f"Res{resid}")
        label = f"{resid}_{resname}"
        label = modify_label(label)
        concatenated_labels.append(label)

        start_time = residue_index * time_per_residue_with_gap
        end_time = start_time + residue_duration
        x_residue = np.linspace(start_time, end_time, len(filtered_times))
        x_list.extend(x_residue)

        for prop in properties:
            density_residue = rrcs_sum_B_properties[resid][prop]
            density_residue_series = pd.Series(density_residue)
            smoothed_density_residue = density_residue_series.rolling(window=args.window_size, min_periods=1, center=True).mean().values
            smoothed_density_data_per_property[prop].extend(smoothed_density_residue)

        gap_start = end_time
        gap_end = gap_start + gap_duration
        len_filtered_times_gap = max(1, int(len(filtered_times) * args.gap_fraction))
        x_gap = np.linspace(gap_start, gap_end, len_filtered_times_gap)
        x_list.extend(x_gap)
        for prop in properties:
            smoothed_density_data_per_property[prop].extend([0] * len_filtered_times_gap)

    # Create combined plot
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Bar and Scatter Plot (Top)
    ax1 = fig.add_subplot(gs[0])
    x_positions_B = np.arange(len(df1_filtered_B))
    bars_B = ax1.bar(x_positions_B, df1_filtered_B['Mean'], yerr=df1_filtered_B['SE'],
                     color='white', edgecolor=df1_filtered_B['Color'], linewidth=2,
                     width=0.65, capsize=5, error_kw=dict(lw=3, capsize=8), label='Mean RRCS')

    jitter_B = np.random.uniform(-args.jitter_strength, args.jitter_strength, size=df2_filtered_B.shape[0])
    ax1.scatter(x_positions_B[np.searchsorted(df1_filtered_B['Modified_Residue'], df2_filtered_B['Modified_Residue'])] + jitter_B,
                df2_filtered_B['Value'], color=df2_filtered_B['Color'], alpha=0.6, s=20, label='Individual RRCS', marker='o')

    ax1.set_ylim(0, max(df2_filtered_B['Value'].max(), df1_filtered_B['Mean'].max() + df1_filtered_B['SE'].max()) * 1.1)
    ax1.set_xticks(x_positions_B)
    ax1.set_xticklabels(df1_filtered_B['Modified_Residue'], rotation=90, fontsize=12)
    ax1.set_ylabel('RRCS Value', fontsize=14)
    ax1.legend(loc='upper left')
    for spine in ['bottom', 'left', 'top', 'right']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(1.5)
    ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, colors='black')
    ax1.grid(False)

    # Density Plot (Bottom)
    ax2 = fig.add_subplot(gs[1])
    x = np.array(x_list)
    bottom_smoothed = np.zeros_like(x)
    for prop in properties:
        density_data = np.array(smoothed_density_data_per_property[prop])
        ax2.fill_between(x, bottom_smoothed, bottom_smoothed + density_data,
                         color=PROPERTY_COLORS[prop], label=prop, alpha=0.7, step='post')
        bottom_smoothed += density_data

    boundary_times = [i * time_per_residue_with_gap for i in range(1, num_residues)]
    for boundary in boundary_times:
        ax2.axvline(x=boundary, color='grey', linestyle='--', linewidth=0.5)

    midpoints = [(i * time_per_residue_with_gap + (i + 1) * time_per_residue_with_gap) / 2 for i in range(num_residues)]
    ax2.set_xticks(midpoints)
    ax2.set_xticklabels(concatenated_labels, rotation=90, fontsize=12)
    ax2.set_ylim(0, 0.25)  # Adjust based on data range
    ax2.set_xlabel('Residues over Time (ns)', fontsize=14)
    ax2.set_ylabel('RRCS Density', fontsize=14)
    ax2.legend(title="Residue Properties", bbox_to_anchor=(1.05, 1), loc='upper left')
    for spine in ['bottom', 'left', 'top', 'right']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(1.5)
    ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, colors='black')
    ax2.grid(False)

    # Save plot
    plt.tight_layout()
    out_filename = os.path.join(args.output_dir, "Combined_RRCS_Visualization.png")
    plt.savefig(out_filename, format='png', transparent=False, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()