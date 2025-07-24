#!/usr/bin/env python3
"""
geonergy.py

Full pipeline (gas‐phase):
  1) Monte Carlo burial sampling (M rotations)
  2) Gas‐phase ΔG sampling
  3) Centroid clustering
  4) Plots A–E, CSV log
  5) [optional] ML regression (RF/LGBM) → ΔG
  6) [optional] classification (Ridge/ElasticNet) → binder vs non-binder
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats            import zscore, pearsonr
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from pdbfixer               import PDBFixer
from openmm.app             import Modeller, ForceField, Simulation, NoCutoff, HBonds
from openmm                 import unit, LangevinIntegrator, OpenMMException

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, StratifiedKFold

# ─── PARAMETERS ─────────────────────────────────────────────────────────────
FF_FILES         = ['amber14-all.xml','gaff.xml']
TEMPERATURE      = 300 * unit.kelvin
FRICTION         = 1.0 / unit.picosecond
DEFAULT_NSTEPS   = 2000
DEFAULT_DT       = 0.001
DEFAULT_INTERVAL = 100
MINIM_STEPS      = 5000
EQUIL_STEPS      = 10000

VdW_R = {
    'H':1.20,'C':1.70,'N':1.55,'O':1.52,'S':1.80,
    'P':1.80,'F':1.47,'Cl':1.75,'Br':1.85,'I':1.98
}

# ─── CORE HELPERS ───────────────────────────────────────────────────────────

def to_label(n:int)->str:
    lbl=''
    while n>0:
        n,r = divmod(n-1,26)
        lbl = chr(ord('A')+r) + lbl
    return lbl

def parse_sph(path:str):
    centers, radii = [], []
    with open(path) as f:
        for L in f:
            parts = L.split()
            if len(parts)>=5 and parts[0].isdigit():
                x,y,z = map(float, parts[1:4])
                centers.append((x,y,z))
                radii.append(float(parts[4]))
    return np.array(centers), np.array(radii)

def read_coords(pdb:str, keep_h:bool):
    pts, elems = [], []
    with open(pdb) as f:
        for L in f:
            if not L.startswith(('ATOM  ','HETATM')): continue
            el = L[76:78].strip().capitalize()
            if el=='H' and not keep_h: continue
            try:
                x,y,z = map(float, (L[30:38], L[38:46], L[46:54]))
            except ValueError:
                continue
            pts.append((x,y,z)); elems.append(el)
    return np.array(pts,float), elems

def compute_metrics(pts, elems, centers, radii):
    N = len(pts)
    if N==0 or centers.size==0:
        return np.zeros(9)
    d2     = np.sum((pts[:,None]-centers[None,:,:])**2, axis=2)
    inside = d2 <= radii[None]**2
    depths = np.clip((radii[None]-np.sqrt(d2))/radii[None], 0,1)
    best_i = int(inside.sum(0).argmax())
    mask   = inside[:,best_i]
    d_best = depths[:,best_i]

    frac    = mask.sum()/N
    depth   = d_best.mean()
    score   = frac * depth

    w       = np.array([VdW_R.get(e,1.5)**3 for e in elems])
    wfrac   = (mask*w).sum()/w.sum()
    wdepth  = (d_best*w).sum()/w.sum()
    wscore  = wfrac * wdepth

    isC     = np.array([e=='C' for e in elems])
    hydro   = (mask & isC).sum()/max(isC.sum(),1)

    uniform = max(0.,1 - d_best.std())
    uscore  = score * uniform

    return np.array([frac,depth,score,
                     wfrac,wdepth,wscore,
                     hydro,uniform,uscore])

def avg_burial(pts, elems, centers, radii, M):
    if M<=1:
        m=compute_metrics(pts, elems, centers, radii)
        return m, np.zeros_like(m)
    ctr = pts.mean(0)
    mats=[]
    for i in range(M):
        pr = pts if i==0 else R.random().apply(pts-ctr)+ctr
        mats.append(compute_metrics(pr, elems, centers, radii))
    A = np.vstack(mats)
    return A.mean(0), A.std(0)

def cluster_poses(centroids, names, cutoff):
    dists = pdist(np.array(centroids))
    Z     = linkage(dists, method='average')
    labs  = fcluster(Z, t=cutoff, criterion='distance')
    return dict(zip(names, labs))

def build_gas_phase_system(topo):
    ff = ForceField(*FF_FILES)
    return ff.createSystem(topo, nonbondedMethod=NoCutoff,
                           constraints=HBonds, rigidWater=True)

def sample_energies(topo, pos, nsteps, dt, interval):
    integ = LangevinIntegrator(TEMPERATURE, FRICTION, dt*unit.picoseconds)
    sim   = Simulation(topo, build_gas_phase_system(topo), integ)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=MINIM_STEPS)
    sim.context.setVelocitiesToTemperature(TEMPERATURE)
    sim.step(EQUIL_STEPS)
    es=[]
    for i in range(1,nsteps+1):
        sim.step(1)
        if i%interval==0:
            e = sim.context.getState(getEnergy=True) \
                   .getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            es.append(e)
    return np.array(es,float)

def compute_binding_deltaG(rec, lig, nsteps, dt, interval, verbose=False):
    fix = PDBFixer(filename=rec)
    fix.findMissingResidues(); fix.findMissingAtoms()
    fix.addMissingAtoms(); fix.addMissingHydrogens(pH=7.0)
    tr, pr = fix.topology, fix.positions
    fix = PDBFixer(filename=lig)
    fix.findMissingResidues(); fix.findMissingAtoms()
    fix.addMissingAtoms(); fix.addMissingHydrogens(pH=7.0)
    tl, pl = fix.topology, fix.positions
    modeller = Modeller(tr, pr); modeller.add(tl, pl)
    tc, pc = modeller.topology, modeller.positions
    if verbose:
        print(f"  • Sampling ΔG for {os.path.basename(lig)} (gas)")
    Ec = sample_energies(tc, pc, nsteps, dt, interval)
    Er = sample_energies(tr, pr, nsteps, dt, interval)
    El = sample_energies(tl, pl, nsteps, dt, interval)
    deltaG = Ec.mean() - Er.mean() - El.mean()
    stderr = np.sqrt(Ec.var(ddof=1)/len(Ec)
                    +Er.var(ddof=1)/len(Er)
                    +El.var(ddof=1)/len(El))
    return float(deltaG), float(stderr)

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sph',            required=True)
    parser.add_argument('--rec',            required=True)
    parser.add_argument('--ligdir',         required=True)
    parser.add_argument('--outdir',         required=True)
    parser.add_argument('--frac_thresh',    type=float, default=0.5)
    parser.add_argument('--depth_thresh',   type=float, default=0.3)
    parser.add_argument('--vicinity_radius',type=float, default=2.0)
    parser.add_argument('--keep_h',         action='store_true')
    parser.add_argument('--nsteps',         type=int,   default=DEFAULT_NSTEPS)
    parser.add_argument('--dt',             type=float, default=DEFAULT_DT)
    parser.add_argument('--interval',       type=int,   default=DEFAULT_INTERVAL)
    parser.add_argument('--nrot',           type=int,   default=1)
    parser.add_argument('--cluster_cutoff', type=float, default=None)
    parser.add_argument('--ml',             action='store_true')
    parser.add_argument('--hpo',            action='store_true')
    parser.add_argument('--verbose',        action='store_true')
    # ← Add classify here, using the same parser variable
    parser.add_argument('--classify',       action='store_true',
                        help="Perform binder vs. non-binder classification")

    args = parser.parse_args()


    os.makedirs(args.outdir, exist_ok=True)
    vdir = os.path.join(args.outdir, 'vicinity_ligs')
    os.makedirs(vdir, exist_ok=True)

    centers, radii = parse_sph(args.sph)
    pdbs          = sorted(glob.glob(os.path.join(args.ligdir, '*.pdb')))
    if args.verbose:
        print(f"Found {len(pdbs)} PDB(s) in {args.ligdir}")
    if not pdbs:
        print("No PDB files; exiting."); return

    kept = []
    for pdb in pdbs:
        pts, elems = read_coords(pdb, args.keep_h)
        if pts.size==0:
            if args.verbose: print(f"Skipping {pdb}: no atoms")
            continue
        centroid = pts.mean(0)
        mean_m, std_m = avg_burial(pts, elems, centers, radii, args.nrot)
        if args.verbose:
            print(f"{os.path.basename(pdb)} → frac={mean_m[0]:.3f}±{std_m[0]:.3f}, "
                  f"depth={mean_m[1]:.3f}±{std_m[1]:.3f}")
        if mean_m[0]<args.frac_thresh or mean_m[1]<args.depth_thresh:
            continue
        try:
            dG, se = compute_binding_deltaG(
                args.rec, pdb, args.nsteps, args.dt, args.interval, args.verbose
            )
        except OpenMMException as e:
            if args.verbose: print(f"Skipping {pdb}: {e}")
            continue
        if not np.isfinite(dG):
            continue
        lbl = to_label(len(kept)+1)
        kept.append((lbl, os.path.basename(pdb), mean_m, std_m, dG, se, centroid))

    if not kept:
        print("No valid poses"); return

    names = [e[1] for e in kept]
    cents = [e[6] for e in kept]
    cl_map = cluster_poses(cents, names, args.cluster_cutoff) if args.cluster_cutoff \
             else {n:1 for n in names}

    uniform_means = np.array([e[2][7] for e in kept])
    zuni          = zscore(uniform_means)

    results = []
    for idx, (lbl, name, m, s, dG, se, cent) in enumerate(kept):
        cid = cl_map[name]
        results.append((cid, lbl, name, *m, *s, zuni[idx], dG, se, cent))

    # write CSV
    cols = [
      'Cluster','Label','Name',
      'Frac_mean','Depth_mean','Score_mean',
      'WFrac_mean','WDepth_mean','WScore_mean',
      'Hydro_mean','Uniform_mean','UScore_mean',
      'Frac_std','Depth_std','Score_std',
      'WFrac_std','WDepth_std','WScore_std',
      'Hydro_std','Uniform_std','UScore_std',
      'ZUniform','ΔG_kcal_per_mol','StdErr'
    ]
    out_csv = os.path.join(args.outdir, 'kept_ligs.csv')
    pd.DataFrame([r[:-1] for r in results], columns=cols).to_csv(out_csv, index=False)

    # ─── TOP-2 SELECTION ───────────────────────────────────────────────────────
    # 1) best by frac & depth
    best_fd = max(results, key=lambda r: (r[3], r[4]))
    # 2) best by energy (different from best_fd if possible)
    sorted_e = sorted(results, key=lambda r: r[21])
    best_e   = sorted_e[0] if sorted_e[0][1]!=best_fd[1] \
               else (sorted_e[1] if len(sorted_e)>1 else sorted_e[0])

    # ─── VICINITY LOG ──────────────────────────────────────────────────────────
    with open(os.path.join(vdir,'vicinity_ligs.log'),'w') as vf:
        vf.write('Label,Name,Distance\n')
        cnt=0
        for r in results:
            lbl,name,*_,cent = r
            dist = np.linalg.norm(cent - best_fd[-1])
            if dist <= args.vicinity_radius:
                vf.write(f"{lbl},{name},{dist:.3f}\n"); cnt+=1
        vf.write(f"Total,{cnt}\n")

    print(f"Done: {len(results)} kept; best_by_FD={best_fd[1]}, best_by_E={best_e[1]}")

    # ─── PLOTS A–E ────────────────────────────────────────────────────────────
    labels  = [r[1] for r in results]
    fracs   = np.array([r[3] for r in results])
    depths  = np.array([r[4] for r in results])
    scores  = np.array([r[5] for r in results])
    wscores = np.array([r[8] for r in results])
    hydros  = np.array([r[9] for r in results])
    zlabels = np.array([r[12] for r in results])
    dg      = np.array([r[21] for r in results])

    # Plot A: Frac vs Depth
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(fracs, depths, c=zlabels, cmap='plasma', s=60, edgecolors='k')
    ax.scatter(best_fd[3], best_fd[4], c='red',   s=150, marker='*', label=f'FD ({best_fd[1]})')
    ax.scatter(best_e[3],  best_e[4],  c='blue',  s=150, marker='o', label=f'E ({best_e[1]})')
    ax.annotate(best_fd[1], (best_fd[3], best_fd[4]), xytext=(5,5), textcoords='offset points', fontweight='bold', color='red')
    ax.annotate(best_e[1],  (best_e[3],  best_e[4]),  xytext=(5,-10), textcoords='offset points', fontweight='bold', color='blue')
    ax.set(xlabel='Frac', ylabel='Depth', title='Frac vs Depth')
    fig.colorbar(sc, ax=ax, label='ZUniform'); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir,'A_frac_depth.png'))

    # Plot B: UniformScore per Pose
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, [r[11] for r in results], edgecolor='k')
    ax.set(xlabel='Pose', ylabel='UniformScore', title='UniformScore per Pose')
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir,'B_uniformscore_bar.png'))

    # Plot C: Score vs WScore
    r_val,_ = pearsonr(scores, wscores)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(scores, wscores, edgecolors='k')
    ax.plot([scores.min(),scores.max()], [scores.min(),scores.max()], '--')
    ax.set(xlabel='Score', ylabel='WScore', title=f'Score vs WScore (r={r_val:.2f})')
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir,'C_score_vs_wscore.png'))

    # Plot D: ZUniform Distribution
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(zlabels, bins=10, edgecolor='k')
    ax.set(xlabel='ZUniform', ylabel='Count', title='ZUniform Distribution')
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir,'D_zuniform_hist.png'))

    # Plot E: Score vs –ΔG
    fig, ax = plt.subplots(figsize=(6,6))
    sc2 = ax.scatter(scores, -dg, s=hydros*200, c=zlabels, cmap='viridis', edgecolors='k')
    ax.scatter(best_fd[5], -best_fd[21], c='red',   s=150, marker='*', label=f'FD ({best_fd[1]})')
    ax.scatter(best_e[5],  -best_e[21],  c='blue',  s=150, marker='o', label=f'E ({best_e[1]})')
    ax.set(xlabel='Score', ylabel='-ΔG (kcal/mol)', title='Score vs Binding Energy')
    fig.colorbar(sc2, ax=ax, label='ZUniform'); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir,'E_score_vs_dG.png'))

    # ─── ML REGRESSION STEP ─────────────────────────────────────────────────────
    if args.ml:
        # Load features & target
        df = pd.read_csv(out_csv)
        feature_cols = [
            'Frac_mean','Depth_mean','Score_mean',
            'WFrac_mean','WDepth_mean','WScore_mean',
            'Hydro_mean','Uniform_mean','UScore_mean'
        ]
        X = df[feature_cols]
        y = df['ΔG_kcal_per_mol']

        # Dynamically choose CV folds
        n_samples = X.shape[0]
        if n_samples < 2:
            print(f"Skipping regression ML: only {n_samples} sample(s)")
        else:
            n_splits = min(5, n_samples)
            print(f"→ regression ML: using {n_splits}-fold CV on {n_samples} samples")
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Pick model: LightGBM if available, else RandomForest
            try:
                from lightgbm import LGBMRegressor
                print("Using LightGBMRegressor")
                model = LGBMRegressor(random_state=42)
            except ImportError:
                from sklearn.ensemble import RandomForestRegressor
                print("Using RandomForestRegressor")
                model = RandomForestRegressor(n_estimators=200, random_state=42)

            # Optional hyperparameter search (only for LightGBM)
            if args.hpo and model.__class__.__name__ == 'LGBMRegressor':
                param_dist = {
                    'n_estimators': [100, 300, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'min_child_samples': [5, 10, 20],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
                search = RandomizedSearchCV(
                    model, param_dist,
                    n_iter=30, cv=cv,
                    scoring='r2', random_state=42, n_jobs=-1
                )
                search.fit(X, y)
                model = search.best_estimator_
                print("HPO best params:", search.best_params_)

            # Cross‐validated R²
            r2s = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
            print(f"ML regression: {n_splits}-fold CV R² = {r2s.mean():.3f} ± {r2s.std():.3f}")

            # Fit on full data, save model
            model.fit(X, y)
            try:
                import joblib
                joblib.dump(model, os.path.join(args.outdir, 'model.joblib'))
                print("Saved model.joblib")
            except ImportError:
                print("joblib not available, skipping save")

            # SHAP summary plot
            try:
                import shap
                explainer   = shap.TreeExplainer(model)
                shap_vals   = explainer(X)
                plt.figure(figsize=(6,4))
                shap.summary_plot(shap_vals, X, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, 'feature_importances_shap.png'))
                print("Saved SHAP plot")
            except ImportError:
                print("shap not available, skipping SHAP plot")


    # ─── CLASSIFICATION STEP ───────────────────────────────────────────────────
    if args.classify:
        from collections import Counter
        from sklearn.linear_model import RidgeClassifier, SGDClassifier

        # 1) Load data and define classes
        df = pd.read_csv(out_csv)
        # e.g. bottom 25% ΔG → “good” binder
        dg_thresh = df['ΔG_kcal_per_mol'].quantile(0.25)
        y_cls = (df['ΔG_kcal_per_mol'] <= dg_thresh).astype(int)

        # 2) Build domain‐informed features
        df['UScore_mean']  = df['Score_mean'] * df['Uniform_mean']
        df['Hydro_wdepth'] = df['Hydro_mean']  * df['WDepth_mean']
        df['Depth_frac']   = df['Depth_mean']  * df['Frac_mean']
        df['log_WFrac']    = np.log1p(df['WFrac_mean'])
        feat_cls = [
            'Frac_mean','Depth_mean','Score_mean',
            'WFrac_mean','WDepth_mean','WScore_mean',
            'Hydro_mean','Uniform_mean','UScore_mean',
            'Hydro_wdepth','Depth_frac','log_WFrac'
        ]
        Xc = df[feat_cls].values

        # 3) Dynamic CV splitting
        n_samples = Xc.shape[0]
        class_counts = Counter(y_cls)
        min_class = min(class_counts.values())
        if n_samples < 2 or min_class < 2:
            print(f"Skipping classification ML: insufficient samples ({n_samples}) or class imbalance {dict(class_counts)}")
        else:
            n_splits = min(5, n_samples, min_class)
            print(f"→ classification ML: using {n_splits}-fold Stratified CV on {n_samples} samples (classes {dict(class_counts)})")
            cvc = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # 4) Evaluate Ridge vs ElasticNet
            clf_ridge = RidgeClassifier(alpha=1.0)
            auc_ridge = cross_val_score(clf_ridge, Xc, y_cls, cv=cvc, scoring='roc_auc', n_jobs=-1).mean()

            clf_enet = SGDClassifier(
                loss='log',
                penalty='elasticnet',
                l1_ratio=0.5,
                alpha=1e-3,
                max_iter=10000,
                random_state=42
            )
            auc_enet = cross_val_score(clf_enet, Xc, y_cls, cv=cvc, scoring='roc_auc', n_jobs=-1).mean()

            print(f"Classification AUCs → Ridge: {auc_ridge:.3f}, ElasticNet: {auc_enet:.3f}")

            # 5) Fit best classifier and report coefficients
            best_clf = clf_ridge if auc_ridge >= auc_enet else clf_enet
            best_clf.fit(Xc, y_cls)
            coefs = best_clf.coef_[0]
            print("Feature coefficients (abs-sorted):")
            for feat, coef in sorted(zip(feat_cls, coefs), key=lambda x: abs(x[1]), reverse=True):
                print(f"  {feat:15s}: {coef:+.4f}")

            # 6) Plot classification coefficients
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(feat_cls, coefs, color='C2', edgecolor='k')
            ax.set_xticklabels(feat_cls, rotation=45, ha='right')
            ax.set_ylabel('Coefficient')
            ax.set_title('Classification Feature Coefficients')
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, 'classification_coeffs.png'))
            print("Saved classification_coeffs.png")

if __name__=='__main__':
    main()
