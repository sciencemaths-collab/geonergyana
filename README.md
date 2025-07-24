[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sciencemaths-collab/geonergyana/blob/main/run_colab.ipynb)

# enegeo_refined4_gas_ml

Initial geometric profiling, gas‑phase binding energy estimates, and ML analysis for receptor–peptide complexes.

**Analyze peptide docking to receptor surface spheres**, combining hydrophobic‑geometry metrics, single‑trajectory ΔG sampling in vacuum (OpenMM), and optional ML regression/classification.

---

## 🚀 Key Features

- **Geometric Analysis**  
  - Occupancy fraction (Frac)  
  - Normalized burial depth (Depth)  
  - Combined shape‑complementarity Score  
- **Physicochemical Scoring**  
  - van der Waals–weighted metrics (WFrac, WDepth, WScore)  
  - Hydrophobic contact fraction (HydroFrac)  
- **Uniformity Metrics**  
  - UniformScore & ZUniform to flag uneven or exceptional binding  
- **Energy Estimation**  
  - Vacuum‑phase ΔG & standard error via single‑trajectory MD  
- **Automated Visualization**  
  - Publication‑quality plots A–E  
- **Machine Learning (optional)**  
  - Regression (LightGBM or RandomForest) predicting ΔG  
  - Hyperparameter optimization (`--hpo`)  
  - Classification (Ridge vs ElasticNet) of binder vs non‑binder  
  - SHAP summary plot of feature importances  
- **Comprehensive Output**  
  - `results/kept_ligs.csv` summary  
  - Vicinity‑distance log for best‑pose context  

---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/sciencemaths-collab/geonergyana.git
   cd geonergyana
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/openmm/pdbfixer.git
   ```

   Your `requirements.txt` should include:

   ```
   numpy
   pandas
   matplotlib
   scipy
   mdtraj
   biopython
   openmm
   scikit-learn
   lightgbm      # optional for ML
   shap          # optional for SHAP plots
   joblib        # optional for saving models
   ```
3. **Prepare inputs**

   * A `.sph` file defining surface spheres
   * A cleaned receptor PDB
   * A directory of pose PDBs

---

## 🎯 Usage

```bash
python enegeo_refined4_gas_ml.py \
  --sph    path/to/fg_surface.sph \
  --rec    path/to/receptor.pdb \
  --ligdir path/to/poses/ \
  --outdir path/to/results/ \
  --frac_thresh    0.5 \
  --depth_thresh   0.3 \
  --vicinity_radius 2.0 \
  --keep_h         \
  --nsteps 2000    \
  --dt     0.001   \
  --interval 100   \
  --ml             \
  --hpo            \
  --classify       \
  --verbose
```

| Flag                | Type  | Default | Description                                                |
| ------------------- | ----- | ------- | ---------------------------------------------------------- |
| `--sph`             | file  | ―       | `.sph` with sphere indices, centers (x,y,z), and radii (Å) |
| `--rec`             | file  | ―       | Receptor PDB                                               |
| `--ligdir`          | dir   | ―       | Directory of pose PDBs                                     |
| `--outdir`          | dir   | ―       | Output folder for CSV, plots, logs                         |
| `--frac_thresh`     | float | 0.5     | Minimum Frac to keep a pose                                |
| `--depth_thresh`    | float | 0.3     | Minimum Depth to keep a pose                               |
| `--vicinity_radius` | float | 2.0     | Å radius for centroid‑to‑best‑pose logging                 |
| `--keep_h`          | flag  | off     | Include hydrogens in geometric metrics                     |
| `--nsteps`          | int   | 2000    | MD steps for ΔG sampling                                   |
| `--dt`              | float | 0.001   | Integrator timestep (ps)                                   |
| `--interval`        | int   | 100     | Record energy every N steps                                |
| `--ml`              | flag  | off     | Perform regression ML                                      |
| `--hpo`             | flag  | off     | Run hyperparameter optimization                            |
| `--classify`        | flag  | off     | Perform binder vs non‑binder classification                |
| `--verbose`         | flag  | off     | Print detailed logs                                        |

---

## 📂 Outputs

* **`results/kept_ligs.csv`**: All kept poses with metrics and ΔG
* **Plots**:

  * `A_frac_depth.png`
  * `B_uniformscore_bar.png`
  * `C_score_vs_wscore.png`
  * `D_zuniform_hist.png`
  * `E_score_vs_dG.png`
* **`results/vicinity_ligs/vicinity_ligs.log`**: Centroid distances to best pose

---

*For issues or questions, please open an issue on GitHub.*

