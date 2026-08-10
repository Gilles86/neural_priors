# Figure notebooks — data inputs

These notebooks build the paper's figures and statistics from **summary TSVs**, not from
the raw BIDS dataset. Small summary files are included in the repo; larger or run-specific
ones must be (re)generated as listed below and placed at the expected relative path
(notebooks are run from `notebooks/`). Behavioral raw data are read from the BIDS folder,
assumed at `../../ds-neuralpriors` relative to this directory (edit `bids_folder` at the
top of each notebook if yours lives elsewhere).

| Expected file | Status | How to (re)generate |
|---|---|---|
| `decoding/fisher_information.tsv` | included | `neural_priors/encoding_model/get_fisher_info.py` (model 31), per-subject TSVs aggregated to `derivatives/fisher_information2/fisher_information.tsv` |
| `decoding/2025-06-26 fisher_information.tsv` | included | same script, earlier run for models 15/18 (snapshot of `derivatives/results_june2025/fisher_information.tsv`) |
| `decoding/expected_uncertainty.tsv` | included | `neural_priors/encoding_model/get_expected_uncertainty.py` → `derivatives/expected_uncertainty/expected_uncertainty.tsv` |
| `main_models_roi-NPCr_desc-groundtruth_parameters 2026-03-04.tsv.gz` | regenerate (~70 MB) | `python neural_priors/encoding_model/write_parameters_summary.py --smoothed` → `derivatives/summary_tsvs/main_models_roi-NPCr_desc-groundtruth_parameters.tsv.gz`; copy here under the dated name (the date marks the snapshot used for the paper) |
| `all_likelihoods.tsv` | regenerate (~16 MB, needs all 14 main models) | `archive/encoding_model/analyze_likelihoods.ipynb` → `derivatives/encoding_models/all_likelihoods.tsv`; copy here |
| `decoding/decoding_pars.tsv` | regenerate | `neural_priors/encoding_model/get_trialwise_neural_measures.ipynb` → `derivatives/decoding2/decoding_pars.tsv`; copy here |
| `stan/outputs/summaries/results_20250909180225.csv` | regenerate | fit `stan/hmodel.stan` with CmdStan on the included `stan/data_for_stan.json` + `stan/inits_stan.json`, then `stansummary --csv_filename=...` (older run summaries are included for reference) |
| `stan/outputs/summaries_fmri_30/…`, `…_fmri_spherical_30/…`, `…_fmri_notspherical_30/…` | regenerate | same model fit on the fMRI-decoded-estimate JSONs built by `stan/make_stan_json_files.ipynb` (which needs `decoding/decoding_pars.tsv`) |

Notebook → input map:

- `fmri_models_analysis.ipynb`: parameter-summary snapshot (`main_models_… .tsv.gz`).
- `behavior_vs_fmri.ipynb`: Stan summary, both Fisher-information TSVs,
  `expected_uncertainty.tsv`, `decoding_pars.tsv`, fMRI-Stan summaries, behavioral data.
- `stan_output.ipynb`: Stan summary (+ fMRI-Stan summary).
- `likelihoods_and_bics.ipynb`: `all_likelihoods.tsv`, parameter-summary snapshot, behavioral data.
- `stan/make_stan_json_files.ipynb`: behavioral data, `decoding/decoding_pars.tsv`.
  **Writes** the Stan input JSONs — see the re-indexing warning in its top cell.
- `participant_info.ipynb`: `participants.tsv` in the BIDS folder (the import cell is a
  documented one-shot — do not re-run).
- `../neural_priors/figures/figure1.ipynb`: behavioral data via the `Subject` class.
