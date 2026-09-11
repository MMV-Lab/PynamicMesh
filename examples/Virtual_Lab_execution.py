"""Driver for the active-surface virtual laboratory on a real cell mesh.

Two material regimes are available (see ActiveSurfaceConstitutiveModel docstring):
  * PRESETS       fluid active surface (paper Sec. III): any shape relaxes to a sphere;
                  use for generic instabilities on spheres/ellipsoids.
  * CELL_PRESETS  viscoelastic active shell (paper Sec. IV) that *remembers the input cell
                  shape*: in-plane elasticity E_shear/E_area, curvature memory, turnover time
                  tau_remodel.  Use these for real cells: deformations stay local and the cell
                  keeps its morphology on time scales shorter than tau_remodel.
  * chemistry     the regulator kinetics are pluggable (ChemistryModel): LinearTurnover (legacy
                  k_turn), MechanosensitiveTurnover (curvature / tension / strain-rate feedback),
                  ExcitableRho (pulsatile or excitable Rho-actomyosin cortex), TuringPolarity
                  (mass-conserved wave-pinning polarity).  Stimuli (optogenetics, drugs) and Events
                  (laser ablation, drug wash-in) reproduce standard perturbation experiments (Sec. 3).
  * protocols     experiments are defined as Protocol building blocks that can be composed in
                  parallel (A + B) or chained in time (A >> B), see Sec. 4.

Output: everything of one execution goes to  Results/Virtual_lab/Experiment_<id>/  (one folder per
lab, one per experiment, GIFs in simulation_gifs/, all console output in log_output.txt); the
console only shows a transient progress bar of the running experiment.
GPU: if CuPy and a CUDA device are present the sparse linear solves run on the GPU (block conjugate
gradient); by default only for meshes with >= 8000 vertices, where it pays off — set
SimulationConfig(gpu="on") to force it, gpu="off" to disable. Without a GPU nothing changes.
Interrupt: press Enter in the console while an experiment runs to stop it after the current step
(its frames/metrics are saved) and continue with the next one (in a sequence: the next stage).
"""
import numpy as np
import pyvista as pv
from pathlib import Path
from PynamicMesh.utils.visualizers import visualize_obj_sequence
from PynamicMesh.core.active_surface import (
    DiscreteGeometryEngine, ActiveSurfaceConstitutiveModel, ActiveSurfaceSimulator,
    VirtualLaboratory, SimulationConfig, prepare_mesh, to_physical_units,
    PRESETS, CELL_PRESETS, CELL_SHELL,
    gaussian_cap, equatorial_ring, noisy_field, uniform_field, LocalNormalForce, AnchorSpring,
    # mechano-chemistry: reaction models, lab-frame stimuli, perturbation events
    ExcitableRho, TuringPolarity, MechanosensitiveTurnover, LinearTurnover,
    GaussianPulse, UniformStimulus, LaserAblation, ParameterStep,
    ExperimentRun, Protocol)
from PynamicMesh.utils.tools import load_aligned_mesh


exp_names = ["01_Cell_Passive_Control", "01b_Fluid_Rounding", "02_Cell_Polar_Contraction", "03_Cell_Cytokinesis_Ring",
             "04_Cell_Torque_Folding", "05_Cell_Protrusion_Anchored", "06_Cell_Buckling"]
others = ["07_cap_E_shear=0.5", "07_cap_E_shear=3", "07_cap_E_shear=10"]
chem_exp_names = ["08_Cell_Pulsatile_Cortex", "09_Cell_Optogenetic_Wave", "10_Cell_Spontaneous_Polarity",
                  "11_Cell_Curvature_Feedback_Folding", "12_Cell_Strain_Feedback_Pulses", "13_Cell_Laser_Ablation",
                  "14_Cell_Ring_Blebbistatin"]
compose_exp_names = ["20_Compose_Protrusion+Buckling", "21_Compose_Protrusion+Pulsatile",
                     "22_Sequence_Protrude-Buckle-Ablate", "23_Sequence_Polarise-then-Divide"]


run = ExperimentRun(root=Path("Results") / "Virtual_lab", echo=False)
print("Output folder:", run.dir)          # the only console print of this script


# Load and prepare the real cell (repair, remesh, quality, unit size R_eq = 1)

mesh_path = Path(r'.\Real_cell\cell_small\surface_1_9.mat')
raw = load_aligned_mesh(mesh_path)
pv_faces = np.c_[np.full(len(raw.faces), 3), raw.faces].flatten()
base_cell = prepare_mesh(pv.PolyData(raw.vertices, pv_faces), target_vertices=4000, verbose=False)
scale = base_cell.field_data["length_scale"][0]
run.log("input mesh:", mesh_path, "| vertices:", base_cell.n_points,
        "| length scale (physical units per simulation unit):", scale)


def run_validation_suite():
    sphere = prepare_mesh(pv.Icosphere(radius=1.0, nsub=4), verbose=False)
    geo = DiscreteGeometryEngine(sphere)
    v, a = geo.get_enclosed_volume_and_area()
    run.log(f"[Geometry] volume err {abs(v - 4/3*np.pi)/(4/3*np.pi):.2e}, area err "
            f"{abs(a - 4*np.pi)/(4*np.pi):.2e}, ∫K dA/4π={(geo.K*geo.A).sum()/(4*np.pi):.6f}")
    sim = ActiveSurfaceSimulator(sphere, ActiveSurfaceConstitutiveModel(kappa_b=1.0, gamma_0=1.0))
    sim.set_initial_chemical_field(np.zeros(geo.n_vertices))
    for _ in range(20):
        dt, m = sim.step_forward(dt_max=0.01)
    run.log(f"[Passive] E={m['Helfrich_Energy']:.4f} (exact {4*np.pi + 8*np.pi:.4f}), "
            f"p={m['Pressure']:.3f} (exact 2), vmax={m['Max_Velocity']:.1e}")


run_validation_suite()


#  Experiments on the real cell (viscoelastic shell = shape is kept, deformations local)

# refine_ratio: edges that stretch beyond 1.6x the initial mean edge 
NUMERICS = SimulationConfig(cfl=0.2, regularize_mesh=0.1, refine_ratio=1.6, max_vertex_factor=2.5,
                            gpu="auto", gpu_min_vertices=8000)   # gpu="on" forces CuPy solves at any size
lab = run.lab("active_surface_lab", save_formats=("obj",), save_fields=True, config=NUMERICS)
L = np.ptp(base_cell.points, axis=0).max()
front = base_cell.points[np.argmax(base_cell.points[:, 0])]
RUN = dict(dt_max=0.01, save_every=10)

# The single experiments are defined as Protocols (building blocks). Regulator fields are given
# as callables (mesh -> field) so that a block can be re-evaluated on a deformed/refined mesh
# when it is composed or chained.
BLOCKS = {
    # Control: the cell should keep its morphology (compare with PRESETS["passive_relaxation"],
    # the fluid surface, which rounds up into a sphere).
    "01_Cell_Passive_Control": Protocol("01_Cell_Passive_Control", CELL_PRESETS["cell_passive"], total_time=1.0),
    "01b_Fluid_Rounding": Protocol("01b_Fluid_Rounding", PRESETS["passive_relaxation"], total_time=1.0),
    # Local contractile cap (myosin patch): cortical flow toward the cap + local flattening
    "02_Cell_Polar_Contraction": Protocol(
        "02_Cell_Polar_Contraction", CELL_PRESETS["cell_contraction"],
        c0=lambda m: gaussian_cap(m, front, 0.3 * L, 1.0, 0.1), total_time=1.5),
    # Contractile ring. For furrow ingression the cortex must remodel faster than the process:
    # tau_remodel ~ 0.5 (cytokinesis-like). tau_remodel=5 keeps the shell elastic (little pinch).
    "03_Cell_Cytokinesis_Ring": Protocol(
        "03_Cell_Cytokinesis_Ring", {**CELL_PRESETS["cell_ring"], "tau_remodel": 0.5},
        c0=lambda m: equatorial_ring(m, (1, 0, 0), 0.12 * L, 1.0, 0.05), total_time=2.0),
    # Active torque patch (apical-constriction-like folding)
    "04_Cell_Torque_Folding": Protocol(
        "04_Cell_Torque_Folding", CELL_PRESETS["cell_torque_folding"],
        c0=lambda m: gaussian_cap(m, front, 0.25 * L, 1.0, 0.0), total_time=1.0),
    # Protrusion driven by polymerisation pressure, opposite side anchored (adhesion)
    "05_Cell_Protrusion_Anchored": Protocol(
        "05_Cell_Protrusion_Anchored", CELL_PRESETS["cell_passive"], total_time=1.0,
        external_forces=[LocalNormalForce(front, strength=3.0, radius=0.2 * L),
                         AnchorSpring(-front, stiffness=5.0, radius=0.3 * L)]),
    # Buckling of a softened cortex under negative active tension
    "06_Cell_Buckling": Protocol(
        "06_Cell_Buckling", CELL_PRESETS["cell_buckling"],
        c0=lambda m: noisy_field(m, 1.0, 0.05), total_time=1.0),
}
for name in exp_names:
    lab.run_protocol(base_cell, BLOCKS[name], **RUN)      # identical to run_experiment(...)

# Sweep: stiffness of the cortex vs. response to the contractile cap
lab.sweep(base_cell, CELL_PRESETS["cell_contraction"], "E_shear", [0.5, 3.0, 10.0],
          c0=gaussian_cap(base_cell, front, 0.3 * L, 1.0, 0.1),
          total_time=1.0, prefix="07_cap", **RUN)

VirtualLaboratory.compare(lab.results, log=run.log, title="Section 2: shell experiments")
run.export_animations(lab.results, exp_names)               # -> <run>/simulation_gifs/<name>.gif

# Physical units: frames are in R_eq units; map back with the stored scale.
final = to_physical_units(lab.results["03_Cell_Cytokinesis_Ring"].trajectory[-1], base_cell)
final.save(str(run.path("ring_final_physical_units.obj")))
# Reload later:  traj = VirtualLaboratory.load_trajectory(run.dir / "active_surface_lab/03_Cell_Cytokinesis_Ring")


# Mechano-chemical experiments: pluggable regulator kinetics, stimuli, events

# Every experiment above used the legacy kinetics (LinearTurnover: c relaxes to c0). The
# ChemistryModel hook closes the feedback loop mechanics -> chemistry -> mechanics and adds
# the standard perturbation assays.  Extra species (rho, u, ...) are saved with the frames.
lab_chem = run.lab("active_surface_lab_chem", save_formats=("obj",), save_fields=True, config=NUMERICS)
# mechanics only (no legacy k_turn / D_chem: the chemistry model owns the kinetics)
MECH = {**CELL_PRESETS["cell_contraction"], "k_turn": 0.0, "D_chem": 0.0}
RUN5 = dict(dt_max=0.01, save_every=5)

CHEM_BLOCKS = {
    # 08  Pulsatile cortex: excitable Rho–actomyosin oscillator, noise nucleates traveling waves
    #     (period ~0.5 time units; look at c_std(t) in metrics.csv and the 'rho' point array).
    "08_Cell_Pulsatile_Cortex": Protocol(
        "08_Cell_Pulsatile_Cortex", MECH, c0=lambda m: uniform_field(m, 0.5), total_time=3.0,
        chemistry=ExcitableRho(noise=0.3, diffusion={"c": 0.002, "rho": 0.02})),
    # 09  Optogenetic activation of an excitable (quiescent) cortex: a 0.1-long light pulse at the
    #     front fires a single contraction wave; the remaining pulses test refractoriness.
    "09_Cell_Optogenetic_Wave": Protocol(
        "09_Cell_Optogenetic_Wave", MECH, c0=lambda m: uniform_field(m, 0.19), total_time=2.0,
        chemistry=ExcitableRho.excitable(),
        stimuli=[GaussianPulse(front, radius=0.3 * L, amplitude=6.0, t_on=0.3, t_off=1.9, period=0.8, duty=0.125)]),
    # 10  Spontaneous polarisation (mass-conserved wave-pinning) reinforced by cortical flow:
    #     a single contractile cap emerges from noise and the flow toward it sharpens it.
    "10_Cell_Spontaneous_Polarity": Protocol(
        "10_Cell_Spontaneous_Polarity", {**MECH, "zeta": 0.8}, c0=lambda m: noisy_field(m, 0.4, 0.15),
        total_time=3.0, chemistry=TuringPolarity(total=1.5)),
    # 11  Curvature sensing (BAR-like recruitment to high H) + active torque bending inward
    #     where c is high: a mechano-chemical folding instability of the cortex.
    "11_Cell_Curvature_Feedback_Folding": Protocol(
        "11_Cell_Curvature_Feedback_Folding",
        {**CELL_PRESETS["cell_torque_folding"], "k_turn": 0.0, "D_chem": 0.0, "zeta_c": -0.3},
        c0=lambda m: noisy_field(m, 1.0, 0.05), total_time=1.0,
        chemistry=MechanosensitiveTurnover(k_turn=2.0, alpha_curvature=1.0, diffusion={"c": 0.01})),
    # 12  Compression-recruits-myosin (Munjal et al. 2015): a purely mechanical positive feedback
    #     (compression -> myosin -> more compression) drives a contractile clustering instability of a
    #     homogeneous cortex seeded by noise; the threshold lies between alpha_compression ~5 and ~10.
    "12_Cell_Strain_Feedback_Pulses": Protocol(
        "12_Cell_Strain_Feedback_Pulses", {**MECH, "eta_s": 0.1}, c0=lambda m: noisy_field(m, 1.0, 0.05),
        total_time=2.0,
        chemistry=MechanosensitiveTurnover(k_turn=1.0, alpha_compression=8.0, diffusion={"c": 0.01}, noise=0.1)),
}
for name in ("08_Cell_Pulsatile_Cortex", "09_Cell_Optogenetic_Wave", "10_Cell_Spontaneous_Polarity",
             "11_Cell_Curvature_Feedback_Folding", "12_Cell_Strain_Feedback_Pulses"):
    lab_chem.run_protocol(base_cell, CHEM_BLOCKS[name], **RUN5)

# 13  Laser ablation of a contractile cortex: recoil speed of the wound margin is the cortical
#     tension read-out; c_eq re-assembles with recovery_time.
ablation = LaserAblation(front, radius=0.15 * L, t_cut=0.3, recovery_time=1.0)
CHEM_BLOCKS["13_Cell_Laser_Ablation"] = Protocol(
    "13_Cell_Laser_Ablation", CELL_PRESETS["cell_contraction"], c0=lambda m: uniform_field(m, 1.0),
    total_time=1.5, events=[ablation])
recoil = []
lab_chem.run_protocol(base_cell, CHEM_BLOCKS["13_Cell_Laser_Ablation"], **RUN5,
                      callback=lambda sim, met: recoil.append((sim.t, sim.events[0].recoil_speed(sim))))
recoil = np.array(recoil)
i = np.argmax(recoil[:, 1])
run.log(f"[13] max recoil speed {recoil[i, 1]:.3f} at t={recoil[i, 0]:.2f} (cut at t=0.3)")
np.savetxt(lab_chem.output_dir / "13_Cell_Laser_Ablation" / "recoil.csv", recoil,
           delimiter=",", header="time,recoil_speed", comments="")

# 14  Drug wash-in during cytokinesis: blebbistatin (zeta -> 0) at t=1.0 stops furrow ingression;
#     furrow radius is recorded through the callback.
CHEM_BLOCKS["14_Cell_Ring_Blebbistatin"] = Protocol(
    "14_Cell_Ring_Blebbistatin", {**CELL_PRESETS["cell_ring"], "tau_remodel": 0.5},
    c0=lambda m: equatorial_ring(m, (1, 0, 0), 0.12 * L, 1.0, 0.05), total_time=2.0,
    events=[ParameterStep(1.0, zeta=0.0)])
furrow = []
lab_chem.run_protocol(base_cell, CHEM_BLOCKS["14_Cell_Ring_Blebbistatin"], **RUN,
                      callback=lambda sim, met: furrow.append((sim.t, sim.furrow_radius((1, 0, 0), 0.05 * L))))
furrow = np.array(furrow)
run.log(f"[14] furrow radius: t=0 {furrow[0, 1]:.3f} -> t=1 {furrow[np.searchsorted(furrow[:, 0], 1.0), 1]:.3f} "
        f"-> t=2 {furrow[-1, 1]:.3f}")

# 15  Sweep the excitability of the cortex (autocatalysis k_a) under a fixed optogenetic pulse
for k_a in [40.0, 60.0, 80.0]:
    lab_chem.run_experiment(base_cell, MECH, uniform_field(base_cell, 0.19), total_time=1.5,
                            exp_name=f"15_opto_k_a={k_a:g}", **RUN5,
                            chemistry=ExcitableRho.excitable(k_a=k_a),
                            stimuli=[GaussianPulse(front, radius=0.3 * L, amplitude=6.0, t_on=0.3, t_off=0.4)])

VirtualLaboratory.compare(lab_chem.results, keys=("c_mean", "c_std", "c_max", "Sphericity",
                                                  "Aspect_Ratio", "Retrograde_Flow"),
                          log=run.log, title="Section 3: mechano-chemistry")
run.export_animations(lab_chem.results, chem_exp_names)


# Composed simulations: the blocks above (or lab.protocols[...] of anything already run)
#    are combined in parallel (A + B) or chained in time (A >> B).  The single experiments
#    stay as they are; composition creates new experiments.
lab_comp = run.lab("active_surface_lab_composed", save_formats=("obj",), save_fields=True, config=NUMERICS)

# 20  Parallel composition: protrusion + anchoring forces of 05 acting on the softened,
#     negatively-tensed cortex of 06 (later block wins on conflicting parameters; the overrides
#     are written to the log).  A + B  ==  Protocol.compose(A, B)
P20 = BLOCKS["05_Cell_Protrusion_Anchored"] + BLOCKS["06_Cell_Buckling"]
lab_comp.run_protocol(base_cell, P20, exp_name="20_Compose_Protrusion+Buckling", **RUN)

# 21  Parallel composition with a chemistry block and explicit parameter control: the pulsatile
#     Rho cortex (08) under the protrusion/anchor forces of 05, with a milder contractility;
#     regulator fields of the parts are combined with max (or 'sum').
P21 = Protocol.compose(lab.protocols["05_Cell_Protrusion_Anchored"], CHEM_BLOCKS["08_Cell_Pulsatile_Cortex"],
                       name="21_Compose_Protrusion+Pulsatile", params={"zeta": 1.0}, c0_mode="max",
                       total_time=2.0)
lab_comp.run_protocol(base_cell, P21, **RUN5)

# 22  Sequential composition on ONE continuous simulation (geometry, regulator and elastic
#     reference are carried over; forces/stimuli/events of each stage are time-shifted):
#     protrude (05) -> let the softened cortex buckle (06) -> ablate the front (13 block).
#     c_mode of a stage: 'set' (default) replaces c by the block's field, 'add' adds it, 'keep'
#     leaves the regulator untouched and only changes mechanics/forces.
seq22 = (BLOCKS["05_Cell_Protrusion_Anchored"].with_(total_time=0.8)
         >> BLOCKS["06_Cell_Buckling"].with_(total_time=1.0)
         >> CHEM_BLOCKS["13_Cell_Laser_Ablation"].with_(total_time=1.0,
                                                        events=[LaserAblation(front, radius=0.15 * L, t_cut=0.2,
                                                                              recovery_time=1.0)]))
res22 = lab_comp.run_sequence(base_cell, seq22, exp_name="22_Sequence_Protrude-Buckle-Ablate", **RUN)
run.log("[22] stages:", [(s["name"], round(s["t_start"], 2), round(s["t_end"], 2)) for s in res22.stages])

# 23  Biology-motivated chain: spontaneous polarisation (10, wave-pinning chemistry) followed by a
#     cytokinesis ring (03) — the ring is applied on top of the polarised cortex ('add') and the
#     chemistry switches back to plain turnover for the division stage.
seq23 = [CHEM_BLOCKS["10_Cell_Spontaneous_Polarity"].with_(total_time=2.0),
         BLOCKS["03_Cell_Cytokinesis_Ring"].with_(total_time=2.0, c_mode="add", chemistry=LinearTurnover(k_turn=1.0))]
res23 = lab_comp.run_sequence(base_cell, seq23, exp_name="23_Sequence_Polarise-then-Divide", **RUN)

VirtualLaboratory.compare(lab_comp.results, log=run.log, title="Section 4: composed experiments")
run.export_animations(lab_comp.results, compose_exp_names)
run.finish()

# Check the meshes version of the simulations
path = ".Results/active_surface_lab_composed" / compose_exp_names[2]
visualize_obj_sequence(path)

