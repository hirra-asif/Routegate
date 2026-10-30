# Captions and rerun instructions for the selector figures

Run `python build_selector_figures.py` from any working directory. The source reads its adjacent `data/` directory and writes the four figures to its adjacent `Figures/` directory. To install regenerated PDFs directly into an existing thesis folder, pass `--output-dir /absolute/path/to/thesis/Figures/posthoc_update_v1`. Requirements: Python 3, matplotlib and numpy. CSVs are frozen audited outcomes; the script performs no training, inference, network requests or judging.

## selector_capacity_profiles

**Selector capacity for the historical two-layer static edit.** Out-of-fold policy outcomes across the three prespecified inner-validation budget profiles compare the linear class selector, linear outcome selector and neural outcome selector. All use the same saved baseline and historical d1 fixed-edit responses (strength 0.15, layers 31–32). E measures retained substantive fixed-edit answers /89; B measures material losses /178; H measures newly harmful fixed-edit responses among 212 H prompts with known non-harmful baselines. One unknown H baseline remains excluded. Unfilled caps and u annotations identify selected unknown B outcomes without adding them to known material-loss counts. Dashed lines mark always-fixed reference outcomes. Profiles constrain observed inner-validation outcomes and do not guarantee outer-fold or deployment safety. At the middle profile the neural selector retains one fewer E answer and incurs one fewer B material loss than the linear outcome selector, with zero H regressions for both.

## moe_selector_profiles

**Outcome selection using residual and native routing features for the frozen three-layer edit.** All nine arm/profile combinations are shown under nested evaluation with the original five development folds. The edit is consensus23 at strength 0.15, layers 31–33. E counts retained substantive static answers /89; B counts material losses /178; H counts harmful static responses selected /213, not newly harmful assistance or complete-system harm. The single E unknown is shown separately where selected. Dashed lines mark the always-fixed inventory. Routing trades eight additional E outcomes and one fewer harmful static response for three additional B material losses at the middle profile. Combined retains one additional E outcome with unchanged B/H counts at the strict profile; no consistent improvement is established across profiles.

## selector_fold_variation

**Outer-fold variation in the number of edits selected.** The top row shows the selector-capacity study; the bottom row shows the MoE feature study. Columns retain the three inner-validation budget profiles. Each point is one observed outer-fold count. The studies use different fold assignments, so equal fold numbers do not identify matched prompt groups across rows. The distribution illustrates variation in complete fitted policies; it cannot attribute that variation uniquely to representation learning, regularisation or threshold choice. No uncertainty interval is inferred from the five plotted folds.

## moe_h_source_controls

**Within-H ranking quality pooled and stratified by source.** AUC and average precision are computed from saved out-of-fold harmful-static prediction scores for the three feature arms. Positive counts and denominators accompany every population. The shaded column contains all 213 H prompts; the remaining columns assess each source separately. Dashed segments in the AP panel show each population's positive fraction, rather than an uncertainty interval. Routing exceeds residual numerically in both AUC and AP within each source, although only five to ten positive outcomes are available in each stratum. Source composition may influence pooled rankings, but these descriptive controls do not establish that it explains the routing advantage. The heads predict judged static-response outcomes; these results do not identify causal safety experts.

## Format checks

Each plot is saved as a vector PDF, an SVG retaining editable text, and a 300 dpi PNG. Colours distinguish endpoints (E copper #B47B48, B blue #2F6595, H rust #9E594C); hatches or marker shapes distinguish methods. No values, labels or intervals are inferred beyond the frozen source CSVs. The source-data SHA-256 digests are recorded in `selector_figure_manifest.json`.
