"""
fluid_analysis.py — Capture grid-level momentum, vorticity and wall flux
for a single genome over one simulation run. Outputs time-series CSV.

Usage:
    uv run python helpers/fluid_analysis.py --aurelia
    uv run python helpers/fluid_analysis.py --genome "[0.15,-0.06,...]"
    uv run python helpers/fluid_analysis.py --gen 42 --log output/cloud/4090/seed_42/evolution_log_seed_42.csv
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JELLY_INSTANCES", "1")

import numpy as np
import taichi as ti
import csv

import mpm_sim as sim
from make_jelly import fill_tank, AURELIA_GENOME, random_genome

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
SAMPLE_EVERY = 200   # substeps between grid snapshots (~100 per actuation cycle)

def sample_grid():
    """Read grid_v and grid_m from GPU; compute fluid momentum, vorticity, wall flux."""
    gv = sim.grid_v.to_numpy()[0]   # (n_grid, n_grid, 2)
    gm = sim.grid_m.to_numpy()[0]   # (n_grid, n_grid)
    n  = sim.n_grid
    dx = 1.0 / n

    fluid_mask = gm > 0

    # Total grid momentum (proxy for bulk fluid motion)
    mom_x = float(np.sum(gv[..., 0] * gm))
    mom_y = float(np.sum(gv[..., 1] * gm))

    # Mean speed of fluid cells
    speed = np.sqrt(gv[..., 0]**2 + gv[..., 1]**2)
    mean_speed = float(np.mean(speed[fluid_mask])) if fluid_mask.any() else 0.0
    max_speed  = float(np.max(speed[fluid_mask]))  if fluid_mask.any() else 0.0

    # Vorticity: curl(v) = dvx/dy - dvy/dx  (central differences)
    dvx_dy = (np.roll(gv[..., 0], -1, axis=1) - np.roll(gv[..., 0], 1, axis=1)) / (2*dx)
    dvy_dx = (np.roll(gv[..., 1], -1, axis=0) - np.roll(gv[..., 1], 1, axis=0)) / (2*dx)
    vort   = dvx_dy - dvy_dx
    mean_vort_mag = float(np.mean(np.abs(vort[fluid_mask]))) if fluid_mask.any() else 0.0
    max_vort_mag  = float(np.max(np.abs(vort[fluid_mask])))  if fluid_mask.any() else 0.0

    # Wall flux — momentum in boundary-normal direction at each wall
    # Top wall (j near n-1): upward velocity hitting ceiling
    damp = n // 20
    top_flux    = float(np.sum(gv[:, -damp:, 1] * gm[:, -damp:]))   # + = toward ceiling
    bottom_flux = float(np.sum(gv[:,  :damp, 1] * gm[:,  :damp]))   # - = toward floor
    left_flux   = float(np.sum(gv[ :damp, :, 0] * gm[ :damp, :]))
    right_flux  = float(np.sum(gv[-damp:, :, 0] * gm[-damp:, :]))

    # Upward vs downward momentum in full domain
    vy = gv[..., 1]
    up_mom   = float(np.sum(vy[vy > 0] * gm[vy > 0]))
    down_mom = float(np.sum(vy[vy < 0] * gm[vy < 0]))

    return dict(
        mom_x=mom_x, mom_y=mom_y,
        mean_speed=mean_speed, max_speed=max_speed,
        mean_vort_mag=mean_vort_mag, max_vort_mag=max_vort_mag,
        top_flux=top_flux, bottom_flux=bottom_flux,
        left_flux=left_flux, right_flux=right_flux,
        up_mom=up_mom, down_mom=down_mom,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aurelia', action='store_true')
    parser.add_argument('--genome', type=str, default=None)
    parser.add_argument('--gen', type=int, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--steps', type=int, default=60000)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.aurelia:
        genome, label = AURELIA_GENOME, 'aurelia'
    elif args.genome:
        genome, label = np.array(json.loads(args.genome)), 'custom'
    elif args.gen is not None:
        log_path = args.log or os.path.join(OUTPUT_DIR, 'best_genomes.json')
        # Try to find best_genomes json near the log
        if args.log:
            import glob
            candidates = glob.glob(os.path.join(os.path.dirname(args.log), 'best_genomes*.json'))
            log_path = candidates[0] if candidates else log_path
        recs = json.load(open(log_path))
        matches = [r for r in recs if r['generation'] == args.gen]
        genome, label = np.array(matches[0]['genome']), f'gen{args.gen}'
    else:
        genome, label = random_genome(), 'random'

    print(f'Genome: {label}  {np.round(genome, 4)}')
    pos, mat, fiber, stats = fill_tank(genome, sim.n_particles)
    sim.load_particles(0, pos, mat, fiber)
    ti.sync()
    sim.sim_time[None] = 0.0

    output_path = args.output or os.path.join(OUTPUT_DIR, f'fluid_analysis_{label}.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = []
    period = 1.0 / sim.actuation_freq

    print(f'Running {args.steps} steps, sampling every {SAMPLE_EVERY}...')
    for step in range(args.steps):
        sim.substep()
        if step % SAMPLE_EVERY == 0:
            t = sim.sim_time[None]
            phase = (t % period) / period
            row = sample_grid()
            row['step'] = step
            row['t'] = t
            row['phase'] = phase
            # Actuation phase label
            if phase < 0.2:
                row['act_phase'] = 'contract'
            elif phase < 0.6:
                row['act_phase'] = 'relax'
            else:
                row['act_phase'] = 'refractory'
            records.append(row)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f'Wrote {len(records)} rows to {output_path}')

    # Quick summary
    import pandas as pd
    df = pd.DataFrame(records)
    print('\n--- Mean by actuation phase ---')
    print(df.groupby('act_phase')[['mom_y','up_mom','down_mom','mean_vort_mag','top_flux','bottom_flux']].mean().round(4))
    print('\n--- Peak wall flux events ---')
    print('Max top_flux (ceiling impact):', df['top_flux'].max())
    print('Min bottom_flux (floor backwash):', df['bottom_flux'].min())
    print('Max vorticity:', df['max_vort_mag'].max())


if __name__ == '__main__':
    main()
