#!/usr/bin/env python3
"""
geonergyana.py

Analyze peptide docking to receptor spheres, focusing on
hydrophobic engagement across all surface spheres, and estimate
binding ΔG for each pose via single-trajectory gas-phase sampling.

Metrics per pose:
  - frac, depth, score, wfrac, wdepth, wscore,
    hydro_frac, uniformity, uniform_score, ZUniform
  - ΔG_kcal_per_mol, StdErr

Generate publication-quality plots A–E and
a CSV log (kept_ligs.csv) with the additional energy columns.

Usage:
    python geonergyana.py \
      --sph fg_surface.sph \
      --rec receptor.pdb \
      --ligdir poses \
      --outdir results \
      [--frac_thresh 0.5] [--depth_thresh 0.3] \
      [--vicinity_radius 2.0] [--keep_h] \
      [--nsteps N] [--dt DT] [--interval I] [--verbose]
"""
import os
import glob
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from pdbfixer import PDBFixer
from openmm.app import Modeller, ForceField, Simulation, NoCutoff, HBonds
from openmm import unit, LangevinIntegrator, OpenMMException

# ─── PARAMETERS ──────────────────────────────────────────────────────────
FF_FILES         = ['amber14-all.xml', 'gaff.xml']
TEMPERATURE      = 300 * unit.kelvin
FRICTION         = 1.0 / unit.picosecond
DEFAULT_NSTEPS   = 2000
DEFAULT_DT       = 0.001
DEFAULT_INTERVAL = 100
MINIM_STEPS      = 5000
EQUIL_STEPS      = 10000

# ─── van der Waals radii ─────────────────────────────────────────────────
VdW_R = {'H':1.20,'C':1.70,'N':1.55,'O':1.52,'S':1.80,
         'P':1.80,'F':1.47,'Cl':1.75,'Br':1.85,'I':1.98}

# ─── ΔG ESTIMATION HELPERS ─────────────────────────────────────────────────
def fix_and_protonate(pdb_path, verbose=False):
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    if verbose:
        print(f"  • PDBFixer applied to {os.path.basename(pdb_path)}")
    return fixer.topology, fixer.positions

def build_gas_phase_system(topology):
    ff = ForceField(*FF_FILES)
    return ff.createSystem(topology,
                           nonbondedMethod=NoCutoff,
                           constraints=HBonds,
                           rigidWater=True)

def sample_energies(topology, positions, nsteps, dt, interval):
    integrator = LangevinIntegrator(TEMPERATURE, FRICTION,
                                    dt * unit.picoseconds)
    system = build_gas_phase_system(topology)
    sim    = Simulation(topology, system, integrator)
    sim.context.setPositions(positions)
    sim.minimizeEnergy(MINIM_STEPS)
    sim.context.setVelocitiesToTemperature(TEMPERATURE)
    sim.step(EQUIL_STEPS)
    energies = []
        for step in range(1, nsteps+1):
        sim.step(1)
        if step % interval == 0:
            # fetch potential energy in one line to avoid indentation errors
            e = sim.context.getState(getEnergy=True)\
                   .getPotentialEnergy()\
                   .value_in_unit(unit.kilocalories_per_mole)
            energies.append(e)
    return np.array(energies, dtype=float)

def compute_binding_deltaG(rec_pdb, lig_pdb, nsteps, dt, interval, verbose=False):
    topo_r, pos_r = fix_and_protonate(rec_pdb, verbose)
    topo_l, pos_l = fix_and_protonate(lig_pdb, verbose)
    modeller = Modeller(topo_r, pos_r)
    modeller.add(topo_l, pos_l)
    topo_c, pos_c = modeller.topology, modeller.positions
    if verbose:
        print(f"  Sampling ΔG for {os.path.basename(lig_pdb)} …")
    E_c = sample_energies(topo_c, pos_c, nsteps, dt, interval)
    E_r = sample_energies(topo_r, pos_r, nsteps, dt, interval)
    E_l = sample_energies(topo_l, pos_l, nsteps, dt, interval)
    deltaG = E_c.mean() - E_r.mean() - E_l.mean()
    stderr = np.sqrt(E_c.var(ddof=1)/len(E_c)
                     + E_r.var(ddof=1)/len(E_r)
                     + E_l.var(ddof=1)/len(E_l))
    return float(deltaG), float(stderr)

# ─── SPHERE-BASED METRICS HELPERS ─────────────────────────────────────────
def to_label(n):
    return chr(ord('A') + n - 1) if n <= 26 else f"Z{n-26}"

def parse_sph(path):
    centers, radii = [], []
    with open(path) as f:
        for l in f:
            parts = l.split()
            if len(parts) >= 5 and parts[0].isdigit():
                x, y, z = map(float, parts[1:4]); r = float(parts[4])
                centers.append((x, y, z)); radii.append(r)
    return np.array(centers), np.array(radii)

def read_coords(path, keep_h=False):
    pts, elems = [], []
    with open(path) as f:
        for l in f:
            if not l.startswith(('ATOM  ', 'HETATM')): continue
            e = l[76:78].strip().capitalize()
            if not keep_h and e == 'H': continue
            try:
                x, y, z = map(float, (l[30:38], l[38:46], l[46:54]))
                pts.append((x, y, z)); elems.append(e)
            except ValueError:
                continue
    return np.array(pts), elems

def compute_metrics(pts, elems, centers, radii):
    N = len(pts)
    if N==0 or centers.size==0:
        return (0.0,) * 9
    d2     = np.sum((pts[:,None] - centers[None,:,:])**2, axis=2)
    inside = d2 <= radii[None]**2
    frac   = float(inside.sum(0).max() / N)
    depths = np.clip((radii[None] - np.sqrt(d2)) / radii[None], 0, 1)
    depth  = float(depths.max(1).mean())
    score  = frac * depth
    w      = np.array([VdW_R.get(e,1.5)**3 for e in elems])
    wfrac  = float((inside * w[:,None]).sum(0).max() / w.sum())
    wdepth = float((depths * w[:,None]).max(1).sum() / w.sum())
    wscore = wfrac * wdepth
    bi      = int(inside.sum(0).argmax())
    C       = np.array([e=='C' for e in elems])
    hydro   = float((inside[:,bi] & C).sum() / max(C.sum(),1))
    atom_d  = depths[:,bi]
    uniform = float(max(0.0, 1 - atom_d.std()))
    uscore  = score * uniform
    return frac, depth, score, wfrac, wdepth, wscore, hydro, uniform, uscore

# ─── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sph', required=True)
    parser.add_argument('--rec', required=True)
    parser.add_argument('--ligdir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--frac_thresh', type=float, default=0.5)
    parser.add_argument('--depth_thresh', type=float, default=0.3)
    parser.add_argument('--vicinity_radius', type=float, default=2.0)
    parser.add_argument('--keep_h', action='store_true')
    parser.add_argument('--nsteps', type=int, default=DEFAULT_NSTEPS)
    parser.add_argument('--dt', type=float, default=DEFAULT_DT)
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    centers, radii = parse_sph(args.sph)
    os.makedirs(args.outdir, exist_ok=True)
    vdir = os.path.join(args.outdir, 'vicinity_ligs')
    os.makedirs(vdir, exist_ok=True)

    pdbs = sorted(glob.glob(os.path.join(args.ligdir, '*.pdb')))
    kept = []
    for pdb in pdbs:
        pts, elems = read_coords(pdb, args.keep_h)
        cent = pts.mean(axis=0) if pts.size else np.zeros(3)
        m    = compute_metrics(pts, elems, centers, radii)

        # Filter by thresholds
        if m[0] < args.frac_thresh or m[1] < args.depth_thresh:
            continue

        # Compute ΔG with error handling
        try:
            dG, se = compute_binding_deltaG(
                args.rec, pdb,
                args.nsteps, args.dt,
                args.interval, args.verbose
            )
        except OpenMMException as e:
            if args.verbose:
                print(f"  • Skipping {os.path.basename(pdb)} due to OpenMM error: {e}")
            continue
        except Exception as e:
            if args.verbose:
                print(f"  • Skipping {os.path.basename(pdb)} due to error: {e}")
            continue

        # Skip non-finite results
        if not np.isfinite(dG):
            if args.verbose:
                print(f"  • Skipping {os.path.basename(pdb)}: ΔG is not finite")
            continue

        # Keep valid pose
        lbl = to_label(len(kept) + 1)
        if args.verbose:
            print(f"{lbl}: {os.path.basename(pdb)} hydro_frac={m[6]:.3f} uniform_score={m[8]:.3f} ΔG={dG:.2f}±{se:.2f} kcal/mol")
        shutil.copy(pdb, args.outdir)
        kept.append((lbl, os.path.basename(pdb), *m, dG, se, cent))

    if not kept:
        print("No valid poses")
        exit()

    # Compute ZUniform and build final list
    uniform_scores = np.array([e[10] for e in kept])
    zuni           = zscore(uniform_scores)

    final = []
    for idx, e in enumerate(kept):
        lbl, name, frac, depth, score, wf, wd, ws, hydro, unif, uscore, dG, stderr, cent = e
        final.append((lbl, name, frac, depth, score, wf, wd, ws, hydro, unif, uscore, zuni[idx], dG, stderr, cent))

    best      = max(final, key=lambda x: x[11])
    best_cent = best[14]

    # Write CSV summary
    out_csv = os.path.join(args.outdir, 'kept_ligs.csv')
    with open(out_csv, 'w') as f:
        headers = ['Idx','Name','Frac','Depth','Score','WFrac','WDepth','WScore','HydroFrac','Uniformity','UniformScore','ZUniform','ΔG_kcal_per_mol','StdErr']
        f.write(','.join(headers) + "\n")
        for e in final:
            f.write(','.join(map(str, e[:14])) + "\n")
        f.write('Best,' + ','.join(map(str, best[:14])) + "\n")

    # Plot A
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter([e[2] for e in final], [e[3] for e in final], c=[e[11] for e in final], cmap='plasma', s=60, edgecolors='k')
    ax.scatter(best[2], best[3], c='red', s=150, marker='*', label='Best')
    for lbl, x, y in zip([e[0] for e in final], [e[2] for e in final], [e[3] for e in final]):
        ax.annotate(lbl, (x,y), textcoords='offset points', xytext=(5,5), fontsize=9)
    ax.set(xlabel='Frac', ylabel='Depth', title='Frac vs Depth')
    fig.colorbar(sc, ax=ax, label='ZUniform')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'A_frac_depth.png'))

    # Plot B
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar([e[0] for e in final], [e[10] for e in final], edgecolor='k')
    ax.set(xlabel='Pose', ylabel='UniformScore', title='UniformScore per Pose')
    fig.tight\layout()
    fig.savefig(os.path.join(args.outdir, 'B_uniformscore_bar.png'))

    # Plot C
    fig, ax = plt.subplots(figsize=(6,6))
    scores = np.array([e[4] for e in final])
    wscores = np.array([e[7] for e in final])
    r_val, _ = pearsonr(scores, wscores)
    ax.scatter(scores, wscores, edgecolors='k')
    ax.plot([scores.min(), scores.max()], [scores.min(), scores.max()], '--')
    ax.set(xlabel='Score', ylabel='WScore', title=f'Score vs WScore (r={r_val:.2f})')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'C_score_vs_wscore.png'))

    # Plot D
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist([e[11] for e in final], bins=10, edgecolor='k')
    ax.set(xlabel='ZUniform', ylabel='Count', title='UniformScore Z-Score Distribution')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'D_zuniform_hist.png'))

    # Plot E
    fig, ax = plt.subplots(figsize=(6,6))
    hydros = np.array([e[8] for e in final])
    scatter = ax.scatter(scores, -np.array([e[12] for e in final]), s=hydros*200, c=[e[11] for e in final], cmap='viridis', edgecolors='k')
    ax.scatter(best[4], -best[12], c='red', s=150, marker='*', label='Best')
    ax.set(xlabel='Score', ylabel='-ΔG (kcal/mol)', title='Score vs Binding Energy')
    fig.colorbar(scatter, ax=ax, label='ZUniform')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'E_score_vs_dG.png'))

    # Vicinity distances
    with open(os.path.join(vdir, 'vicinity_ligs.log'), 'w') as vf:
        vf.write('Idx,Name,Distance\n')
        for lbl,name,*_,cent in final:
            dist = np.linalg.norm(np.array(cent) - best_cent)
            vf.write(f"{lbl},{name},{dist:.3f}\n")
        vf.write(f"Total,{len(final)}\n")

    print(f"Done: {len(kept)} kept; best={best[0]} ZU={best[11]:.2f}")
