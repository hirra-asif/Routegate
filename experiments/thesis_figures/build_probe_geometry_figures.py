#!/usr/bin/env python3
"""Draw additional Experiment 2 figures from saved, verified outputs.

Run from any directory. Requires Python 3.6+ syntax, NumPy and Matplotlib.
No model loading, classifier fitting, network access or evaluation changes.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

BLUE = '#2F6595'
TEAL = '#2A97A4'
AQUA = '#95CDD1'
COPPER = '#B47B48'
RUST = '#9E594C'
INK = '#283940'
MUTED = '#617780'
GRID = '#E3EBEE'
PALE = '#F1F6F7'
DISPLAY_STATES = [8, 16, 24, 32, 40, 45, 48]


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 300,
        'axes.unicode_minus': False,
    })


def auc_pairwise(y, score):
    # E is positive. Ties contribute half, exactly as in ROC-AUC.
    a = score[y == 1, None]
    b = score[y == 0][None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


def reflected_density(values, grid, bandwidth=0.055):
    """Fixed-bandwidth Gaussian smoothing, reflected at the score bounds."""
    values = np.asarray(values)
    centres = np.concatenate([values, -values, 2.0 - values])
    z = (grid[:, None] - centres[None, :]) / bandwidth
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (
        len(values) * bandwidth * np.sqrt(2.0 * np.pi))


def score_distributions(scores):
    fig, ax = plt.subplots(figsize=(5.1, 6.7))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.75, bottom=0.20)
    fig.text(0.02, 0.975, 'E/B probe scores across layers',
             size=12, weight='bold', va='top')
    fig.text(0.02, 0.925,
             'Predictions for prompts held out from each classifier.\nStored repeat 0; five folds.',
             size=8.5, color=MUTED, va='top', linespacing=1.5)
    grid = np.linspace(0.0, 1.0, 401)
    curves = {}
    for state in DISPLAY_STATES:
        for label in ('B', 'E'):
            values = np.array([float(r['predicted_probability_E']) for r in scores
                               if int(r['state']) == state and r['label'] == label])
            curves[state, label] = (values, reflected_density(values, grid))
    highest = max(v[1].max() for v in curves.values())
    rng = np.random.RandomState(20260909)
    stats = []
    for index, state in enumerate(DISPLAY_STATES):
        centre = len(DISPLAY_STATES) - 1 - index
        ax.axhline(centre, color=GRID, linewidth=0.7, zorder=0)
        for label, colour, sign in [('B', BLUE, 1), ('E', COPPER, -1)]:
            values, density = curves[state, label]
            height = 0.39 * density / highest
            ax.fill_between(grid, centre, centre + sign * height,
                            facecolor=colour, alpha=0.28, linewidth=0)
            ax.plot(grid, centre + sign * height, color=colour, linewidth=1.2)
            # Jitter separates symbols visually; it has no data meaning.
            offset = sign * rng.uniform(0.015, 0.095, size=len(values))
            ax.scatter(values, centre + offset, s=5, color=colour,
                       alpha=0.38, linewidths=0, zorder=3)
            stats.append({'state': state, 'layer': state - 1, 'label': label,
                          'n': len(values), 'median_score': float(np.median(values))})
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, len(DISPLAY_STATES) - 0.45)
    ax.set_yticks(np.arange(len(DISPLAY_STATES)))
    ax.set_yticklabels(['Layer ' + str(s - 1) for s in DISPLAY_STATES[::-1]])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel('E probe score (higher values favour E)', labelpad=8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    handles = [Line2D([0], [0], color=BLUE, linewidth=4, label='B: answered benign (178)'),
               Line2D([0], [0], color=COPPER, linewidth=4, label='E: over-refused benign (89)')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.17, 0.855),
               ncol=1, borderaxespad=0, fontsize=8.5)
    fig.text(0.02, 0.082,
             'Same density scale; dots are stored scores.\n'
             'Class-weighted scores are not calibrated probabilities.\n'
             'Layer indices start at 0; layer 44 produces state 45.',
             size=8, color=MUTED, va='top', linespacing=1.5)
    return fig, stats


def direction_stability(rows, weights):
    good = [r for r in rows if int(r['state']) > 0]
    layers = np.array([int(r['state']) - 1 for r in good])
    mean = np.array([float(r['mean_pairwise_cosine_between_folds']) for r in good])
    minimum = np.array([float(r['min_pairwise_cosine_between_folds']) for r in good])
    centroid = np.array([float(r['cosine_with_centroid_difference']) for r in good])
    unit = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    matrix = np.clip(unit @ unit.T, -1.0, 1.0)
    fig = plt.figure(figsize=(10.4, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.38, 1],
                          left=0.07, right=0.96, top=0.72, bottom=0.29, wspace=0.34)
    ax = fig.add_subplot(grid[0, 0])
    hm = fig.add_subplot(grid[0, 1])
    fig.text(0.045, 0.96, 'How consistently do fitted probes point in the same direction?',
             size=14, weight='bold', va='top')
    fig.text(0.045, 0.89, 'Five training folds from repeat 0; coefficients expressed in the original activation units.',
             size=9, color=MUTED, va='top')
    ax.plot(layers, mean, color=TEAL, linewidth=2, label='Mean across fold pairs')
    ax.plot(layers, minimum, color=TEAL, linestyle='--', linewidth=1.2,
            alpha=0.7, label='Smallest fold-pair value')
    ax.plot(layers, centroid, color=COPPER, linewidth=1.6,
            label='Mean probe vs E/B mean difference')
    ax.axvline(44, color=MUTED, linewidth=0.8, linestyle=':', zorder=0)
    ax.set(xlim=(0, 47), ylim=(0, 1.04), xlabel='Layer (zero-based)', ylabel='Signed cosine similarity')
    ax.set_xticks([0, 8, 16, 24, 32, 40, 47])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_title('(a) Direction agreement across layers', pad=14)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.005, -0.30),
              fontsize=8.5, handlelength=2.4, borderaxespad=0)
    cmap = LinearSegmentedColormap.from_list('signed_agreement', [COPPER, '#F8FAFA', BLUE])
    # Vector cells keep the heatmap editable in SVG/PDF, unlike an embedded image.
    edges = np.arange(6) - 0.5
    im = hm.pcolormesh(edges, edges, matrix, cmap=cmap, vmin=-1, vmax=1,
                      shading='flat', rasterized=False)
    hm.set_xlim(-0.5, 4.5)
    hm.set_ylim(4.5, -0.5)
    hm.set_aspect('equal')
    hm.set_title('(b) Layer 44: five fitted probes', pad=14)
    hm.set_xticks(range(5))
    hm.set_yticks(range(5))
    hm.set_xticklabels(range(1, 6))
    hm.set_yticklabels(range(1, 6))
    hm.set_xlabel('Probe fit')
    hm.set_ylabel('Probe fit')
    for i in range(5):
        for j in range(5):
            hm.text(j, i, '{:.2f}'.format(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > 0.65 else INK, fontsize=9)
    for spine in hm.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=hm, fraction=0.048, pad=0.05, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    upper = matrix[np.triu_indices(5, 1)]
    fig.text(0.59, 0.13, 'Layer 44 pairwise mean: {:.3f}\nMinimum: {:.3f}; maximum: {:.3f}'.format(
        upper.mean(), upper.min(), upper.max()), fontsize=9, color=MUTED, linespacing=1.5)
    fig.text(0.045, 0.03,
             '+1: same direction; 0: perpendicular; -1: opposite. Training sets overlap; agreement does not establish a causal steering direction.',
             fontsize=8.2, color=MUTED)
    return fig, matrix


def centroid_geometry(rows):
    good = [r for r in rows if int(r['state']) > 0]
    layers = np.array([int(r['state']) - 1 for r in good])
    distance = np.array([float(r['centroid_distance']) for r in good])
    spread = np.array([float(r['pooled_within_class_sd']) for r in good])
    ratio = np.array([float(r['standardised_distance']) for r in good])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.71, bottom=0.28, wspace=0.35)
    fig.text(0.04, 0.96, 'Separation of the two group means in the original hidden-state space',
             size=13, weight='bold', va='top')
    fig.text(0.04, 0.88, 'Descriptive calculation on all 267 E/B development prompts; no projection or held-out test.',
             size=9, color=MUTED, va='top')
    a, b = axes
    a.plot(layers, spread, color=BLUE, linewidth=2, label='Within-group spread')
    a.plot(layers, distance, color=COPPER, linewidth=2, label='Distance between group means')
    a.set_yscale('log')
    a.set_ylabel('Euclidean distance (log scale)')
    a.set_title('(a) Distance and spread both change', pad=14)
    a.legend(loc='upper left', bbox_to_anchor=(-0.01, -0.29), fontsize=8.5, borderaxespad=0)
    b.plot(layers, ratio, color=TEAL, linewidth=2)
    b.fill_between(layers, 0, ratio, color=TEAL, alpha=0.12)
    b.set_ylim(0, max(0.7, ratio.max() * 1.16))
    b.set_ylabel('Mean distance / within-group spread')
    b.set_title('(b) Separation relative to spread', pad=14)
    for ax in axes:
        ax.set_xlim(0, 47)
        ax.set_xticks([0, 8, 16, 24, 32, 40, 47])
        ax.set_xlabel('Layer (zero-based)')
        ax.grid(axis='y', which='major', color=GRID, linewidth=0.7)
    fig.text(0.54, 0.11,
             'The ratio accounts for changing vector scale.\nIt is descriptive, not a prediction score or causal effect.',
             size=8.6, color=MUTED, linespacing=1.5)
    fig.text(0.04, 0.035,
             'Within-group spread is the square root of the pooled sum of coordinate variances. See the supplied caption for the definition.',
             size=8.3, color=MUTED)
    return fig


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=here / 'inputs_probe_geometry')
    parser.add_argument('--output-dir', type=Path,
                        default=here.parents[1] / 'outputs/thesis_results/experiment_1_2')
    args = parser.parse_args()
    inputs, out = args.input_dir, args.output_dir
    manifest = json.loads((inputs / 'input_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(inputs / name) != expected:
            raise ValueError('Figure input checksum mismatch: ' + name)
    scores = read_csv(inputs / 'raw_probe_scores_repeat0.csv')
    folds = read_csv(inputs / 'cv_assignments.csv')
    fold_map = {r['row_id']: r for r in folds if r['repeat'] == '0'}
    if len(fold_map) != 267:
        raise ValueError('Expected 267 unique prompts in repeat 0.')
    metrics = {(int(r['state']), r['variant'], r['repeat']): r
               for r in read_csv(inputs / 'state_metrics.csv')}
    aucs = {}
    for state in range(1, 49):
        selected = [r for r in scores if int(r['state']) == state]
        if len(selected) != 267 or len({r['row_id'] for r in selected}) != 267:
            raise ValueError('Missing or repeated prediction at state ' + str(state))
        for r in selected:
            f = fold_map[r['row_id']]
            if r['fold'] != f['fold'] or r['label'] != f['final_ebh']:
                raise ValueError('Prediction/fold alignment mismatch.')
        y = np.array([int(r['y_true']) for r in selected])
        score = np.array([float(r['predicted_probability_E']) for r in selected])
        if int(y.sum()) != 89 or not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
            raise ValueError('Unexpected labels or scores.')
        actual = auc_pairwise(y, score)
        expected = float(metrics[state, 'raw', '0']['roc_auc'])
        if abs(actual - expected) > 1e-12:
            raise ValueError('Stored score / AUC mismatch at state ' + str(state))
        aucs[state] = actual
    direction_rows = read_csv(inputs / 'probe_direction_similarity.csv')
    coef = sorted(read_csv(inputs / 'probe_directions_headline_state.csv'), key=lambda r: int(r['fold']))
    if [(r['state'], r['repeat'], r['fold']) for r in coef] != [('45', '0', str(i)) for i in range(5)]:
        raise ValueError('Unexpected probe coefficient rows.')
    weights = np.array([[float(r['w' + str(j)]) for j in range(2048)] for r in coef])
    centroids = read_csv(inputs / 'centroid_separation.csv')
    for r in centroids:
        if int(r['state']) == 0:
            continue
        ratio = float(r['centroid_distance']) / float(r['pooled_within_class_sd'])
        if abs(ratio - float(r['standardised_distance'])) > 1e-12:
            raise ValueError('Centroid ratio mismatch.')
    style()
    score_fig, stats = score_distributions(scores)
    direction_fig, matrix = direction_stability(direction_rows, weights)
    reference = next(r for r in direction_rows if r['state'] == '45')
    upper = matrix[np.triu_indices(5, 1)]
    for actual, column in [(upper.mean(), 'mean_pairwise_cosine_between_folds'),
                            (upper.min(), 'min_pairwise_cosine_between_folds')]:
        if abs(actual - float(reference[column])) > 1e-12:
            raise ValueError('Reconstructed direction similarity mismatch.')
    figures = [('exp2_probe_score_distributions', score_fig),
               ('exp2_probe_direction_stability', direction_fig),
               ('exp2_centroid_geometry', centroid_geometry(centroids))]
    for fmt in ('svg', 'pdf', 'png', 'figure_data'):
        (out / fmt).mkdir(parents=True, exist_ok=True)
    preview_buffer = io.BytesIO()
    with PdfPages(preview_buffer) as preview:
        for name, fig in figures:
            for fmt in ('svg', 'pdf', 'png'):
                buffer = io.BytesIO()
                fig.savefig(buffer, format=fmt)
                (out / fmt / (name + '.' + fmt)).write_bytes(buffer.getvalue())
            preview.savefig(fig)
            plt.close(fig)
    (out / 'Probe_Geometry_Preview.pdf').write_bytes(preview_buffer.getvalue())
    checks = {
        'all_checks_passed': True, 'new_model_runs': 0, 'new_probe_fits': 0,
        'input_files_verified': len(manifest['files']), 'unique_E_B_prompts': 267,
        'counts': {'E': 89, 'B': 178}, 'repeat': 0,
        'held_out_score_rows_verified_against_fold_assignments': len(scores),
        'repeat0_auc_matches_stored_metrics_at_all_48_states': aucs,
        'displayed_states': DISPLAY_STATES, 'displayed_layers': [s - 1 for s in DISPLAY_STATES],
        'curve_bandwidth': 0.055, 'boundary_handling': 'reflection at 0 and 1',
        'score_plot_jitter_seed': 20260909,
        'direction_similarity_state45': {
            'mean': float(upper.mean()), 'min': float(upper.min()), 'max': float(upper.max()),
            'signed': True, 'raw_activation_units': True, 'n_folds': 5, 'repeat': 0},
        'plot_use': 'Descriptive development diagnostics; no new generalisation or causal claim.',
        'label_note': 'Frozen corrected E/B labels are retained; no later response re-labelling is applied.',
    }
    (out / 'probe_geometry_numerical_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    with (out / 'figure_data/probe_score_display_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    np.savetxt(str(out / 'figure_data/probe_direction_cosines_state45_repeat0.csv'),
               matrix, delimiter=',', header='fold_0,fold_1,fold_2,fold_3,fold_4', comments='')
    print('Created three new figures in SVG, PDF and PNG: ' + str(out))
    print('Verified all 12,816 saved scores, 48 AUCs, direction cosines and centroid ratios.')


if __name__ == '__main__':
    main()
