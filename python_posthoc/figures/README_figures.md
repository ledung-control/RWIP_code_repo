# Figures of the revised manuscript (IJIES 20266940)

- `manuscript_figures/Fig1..Fig7_final.png`: the seven figures as inserted in the manuscript (600 dpi; 8 cm or 17 cm wide; serif text >= 11 pt; no panel letters, no outer border).
- `make_fig6.py`: Fig. 6 from `results/runs_600s.csv` and `results/bias_injection_sweep.csv`.
- `make_fig7_traj.py`: 600 s trajectories of the 15 stored policies (BC+TD3 raw, BC-only, BC+TD3 with pi(0) subtracted; seeds 1-5) with the Python re-implementation (`phlib.py`, `actorx.py`); writes `results/fig7_traj.pkl` (check: median accumulated IAE_phi at 600 s = 1144.59 / 89.63 / 5.55, Table 11). Needs `rl_v13_repro/agent_*.mat` in ROOT (about 2.5 min).
- `make_fig7.py`: Fig. 7 from `results/fig7_traj.pkl`.
- `draw_fig1.py` ... `draw_fig5.py`, `fig_common.py`: original drawing scripts for Figs. 1-5 (Fig. 1, 3, 4 diagrams; Fig. 2 membership functions/surface; Fig. 5 nominal responses). The manuscript versions were produced from these with the text size raised to 11 pt, the panel captions/titles removed and the output at 600 dpi; the exact edited copies were not archived, so the PNG files above are the reference.
Run with python3 make_fig6.py etc. from this folder (NumPy, SciPy, pandas, Matplotlib; Liberation Serif font optional).

- `make_final_figures.py` + `masters_submitted/`: generates the submitted Figs. 1-4 PNG files from the archived masters.
- `draw_fig5.py --manuscript`: Fig. 5 in the manuscript style from `results/archived_tables4-7/Compare_v13_results.mat`.
