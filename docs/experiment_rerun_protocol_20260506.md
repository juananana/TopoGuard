# TopoGuard Experiment Rerun Protocol

Date: 2026-05-06

## Backup

Old paper-facing outputs and data were backed up to:

- `backups/experiments_20260506_135522`

The previous paper numbers mainly came from:

- Water QA: `outputs/overall_water_qa_500ep`
- Storm Surge / task2: `outputs/overall_task2_sim`
- Extra diagnostics: `outputs/extra_experiments`

## New Source Of Truth

All rerun results for the revised paper should come from new directories only:

- `outputs/rerun_20260506_water_qa_final`
- `outputs/rerun_20260506_task2`
- `outputs/rerun_20260506_extra`

Do not mix values from historical `fixed`, `final`, `new`, `v2`, or old `500ep` directories.

## Protocol

- Evaluation setting: frozen-profile offline replay over generated/existing execution traces.
- Training profiles are estimated from training records only.
- Final reported quality/cost/latency are measured from held-out test records.
- Main strategy comparison uses profile-based selection under hard normalized cost/latency filters.
- Local repair selection uses frozen training-profile estimates only. The test-side realized quality is used only after the repair candidate has been selected.
- Repair remains a bounded failure-conditioned diagnostic; avoid describing it as a fully deployed online recovery system unless validated separately.
- Storm Surge/task2 transfer is auxiliary evidence because test contexts are noise-perturbed repeats and come from the same project data ecosystem.

## Commands

```powershell
python experiment_overall.py --domain water_qa --train_samples 9 --test_episodes 30 --seed 42 --output outputs/rerun_20260506_water_qa_final
python experiment_overall.py --domain water_qa --reuse --sensitivity --drift --ablation --tau-sensitivity --seed 42 --output outputs/rerun_20260506_water_qa_final
python experiment_overall.py --domain task2 --task2-test-repeats 6 --seed 42 --output outputs/rerun_20260506_task2
python extra_experiments.py --exp all --data outputs/rerun_20260506_water_qa_final --output outputs/rerun_20260506_extra
python plot_paper_v2.py --wqa outputs/rerun_20260506_water_qa_final --task2 outputs/rerun_20260506_task2
```

The v2 plotting script writes figures to `outputs/rerun_20260506_water_qa_final/figures_paper_v2`.
