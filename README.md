# geonergyana
Initial geometric profile and binding energy estimate of a receptor-peptide complex.

**Analyze peptide docking to receptor surface spheres**, focusing on hydrophobic interaction geometry and estimating binding ΔG via single‑trajectory gas‑phase sampling with OpenMM.

---

##  Key Features

- **Geometric Analysis**  
  - Occupancy fraction (Frac)  
  - Normalized burial depth (Depth)  
  - Combined shape‑complementarity Score
- **Physicochemical Scoring**  
  - van der Waals–weighted metrics (WFrac, WDepth, WScore)  
  - Hydrophobic contact fraction (HydroFrac)
- **Uniformity Metrics**  
  - UniformScore & ZUniform to flag uneven or exceptional binding patterns
- **Energy Estimation**  
  - Vacuum‑phase binding free energy (ΔG) & standard error via single‑trajectory MD
- **Automated Visualization**  
  - Five publication‑quality plots (A–E)
- **Comprehensive Output**  
  - `kept_ligs.csv` summary  
  - Vicinity‑distance log for best‑pose context

---

## ⚙ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/sciencemaths-collab/geonergyana.git
   cd geonergyana
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place your input files**:
   - A `.sph` file defining surface spheres  
   - A cleaned receptor PDB  
   - A directory of pose PDBs

---

##  Usage

```bash
geonergy_ana \
  --sph   path/to/fg_surface.sph \
  --rec   path/to/receptor.pdb \
  --ligdir path/to/poses/ \
  --outdir path/to/results/ \
  --frac_thresh    0.5 \
  --depth_thresh   0.3 \
  --vicinity_radius 2.0 \
  --keep_h         \
  --nsteps 2000    \
  --dt     0.001   \
  --interval 100   \
  --verbose
```

###  Arguments

| Flag                | Type      | Default | Description                                                       |
|---------------------|-----------|---------|-------------------------------------------------------------------|
| `--sph`             | `file`    | ―       | `.sph` file with sphere indices, centers (x,y,z) and radii (Å)    |
| `--rec`             | `file`    | ―       | Receptor PDB for analysis                                         |
| `--ligdir`          | `dir`     | ―       | Directory containing pose PDB files                               |
| `--outdir`          | `dir`     | ―       | Output folder for CSV, plots, and logs                            |
| `--frac_thresh`     | `float`   | `0.5`   | Minimum Frac to keep a pose                                       |
| `--depth_thresh`    | `float`   | `0.3`   | Minimum Depth to keep a pose                                      |
| `--vicinity_radius` | `float`   | `2.0`   | Radius (Å) for centroid‐to‐best‐pose distance logging             |
| `--keep_h`          | `flag`    | `off`   | Include hydrogens in geometric metrics                            |
| `--nsteps`          | `int`     | `2000`  | MD sampling steps for ΔG estimation                               |
| `--dt`              | `float`   | `0.001` | Integrator timestep (ps)                                          |
| `--interval`        | `int`     | `100`   | Record energy every N steps                                       |
| `--verbose`         | `flag`    | `off`   | Print detailed logs                                               |

---

##  Outputs

- **`results/kept_ligs.csv`**: Summary of all kept poses with metrics and ΔG   
- **Plots**: `A_frac_depth.png`, `B_uniformscore_bar.png`, `C_score_vs_wscore.png`, `D_zuniform_hist.png`, `E_score_vs_dG.png`  
- **`results/vicinity_ligs/vicinity_ligs.log`**: Distances of each pose’s centroid to best pose

---

*For questions or issues, please open an issue on GitHub.*


