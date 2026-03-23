# cloud_gpu Branch — Session Notes

## Hardware
- Instance: Vast.ai RTX 4090 (24.5 GB VRAM, 128 SMs)
- SSH: `ssh -p 41771 root@72.19.32.135 -L 8080:localhost:8080`
- Project path on instance: `~/jellys/`
- Setup: `bash setup_vastai.sh` (handles uv, deps, Taichi CUDA verification)

## Key Findings

### GPU Bottleneck
- SM utilization: 100% at λ=16, memory only 5.8% used
- Running two parallel `evolve.py` processes time-slices the GPU → same throughput, 2× wall clock
- Fix: single process with larger λ. Scaled to **λ=48** (~244s/gen vs ~72s at λ=16, 3.4× overhead)

### Actuation Strength (was 10000, now 500)
- `actuation_strength=10000` is ~14× E_jelly (700 Pa) → catastrophic instability, morphologies shred
- Swept 8 strengths simultaneously (50→6400) using per-instance `instance_actuation` field
- **400–800 range**: visible stable bell contraction, clean vortex rings jetting downward, bell relaxes cleanly
- **Chosen value: 500**
- Stress is NOT isotropic — `fd.outer_product(fd)` is rank-1 dyadic along fiber tangent ✓
- Sign is correct: tensile stress along bell wall tangent = circumferential contraction = jet downward = jellyfish rises

### Refractory Period Added
- Old waveform: 20% contraction / 80% relaxation → bell oscillated before next stroke
- New waveform: **20% contraction / 40% relaxation / 40% refractory** (activation=0)
- Refractory period lets elastic bell damp oscillations before next stroke fires

### Spawn Height (was 0.7, now 0.4)
- At spawn=0.7, best individuals reached Alt=0.929–0.930 (ceiling threshold=0.93)
- Evolution was converging toward "hit ceiling fastest" not "swim efficiently"
- Fix: `DEFAULT_SPAWN = [0.5, 0.40]` → 0.53 headroom above before ceiling detection
- Water particles fill the same 80K budget, just more above the robot

### Water Model
- Inviscid weakly-compressible MPM (mu=0, lambda=100000)
- Vortex rings confirmed visible in rendered output — correct momentum transfer
- Limitations: 2D geometry (not axisymmetric), numerical viscosity from grid diffusion
- Adequate for fitness landscape; not accurate for drag coefficient quantification

### Infrastructure Added
- `setup_vastai.sh`: full GPU instance provisioning script (CUDA check, uv, deps, Taichi verify)
- `tune_actuation.py`: per-instance actuation sweep — renders N strengths simultaneously in one video
- `per_instance_actuation` field in mpm_sim.py: enables per-instance strength at kernel level

## Current Run
- λ=48, strength=500, spawn=0.40, refractory waveform
- tmux session: `evolve500` on Vast.ai instance
- Log: `output/run_500_spawn04.log`
- Monitor: `ssh -p 41771 root@72.19.32.135 "tail -f ~/jellys/output/run_500_spawn04.log"`
- Rsync output: `rsync -az -e "ssh -p 41771" root@72.19.32.135:~/jellys/output/<run_id>/ ./output/<run_id>/`

## Fitness Signal Status
- Passive baseline drift = +0.062 (jelly structure slightly net-positive buoyant — known calibration issue)
- λ=16 at strength=10000: evolved fitnesses BELOW passive baseline → actuation was catastrophic
- λ=48 at strength=500 (before spawn fix): best fitness +0.20 (3× above baseline) ← first sign of real swimming
- After spawn fix: ceiling no longer caps fitness, expect cleaner fitness landscape

## Remaining Issues
1. High invalid rate (23/48 at gen 3) — self-intersecting genomes polluting CMA-ES signal
2. Passive buoyancy drift (+0.062) still dominates early fitness — may need payload gravity recalibration
3. 2D geometry misses true vortex ring physics (axisymmetric would be more accurate)
