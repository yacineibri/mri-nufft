# %% UTILS FOR TRAJECTORY DISPLAY

# Internal imports
from mrinufft import (
    display_2D_trajectory,
    display_3D_trajectory,
    displayConfig,
    display_gradients_simply,
)
from mrinufft.trajectories.utils import KMAX
import mrinufft.trajectories.tools as tools

import mrinufft as mn
from mrinufft.trajectories.utils import Acquisition, compute_gradients_and_slew_rates, Gammas


# External
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
def show_trajectory(trajectory, one_shot, figure_size):
    if trajectory.shape[-1] == 2:
        ax = display_2D_trajectory(
            trajectory, figsize=figure_size, one_shot=one_shot % trajectory.shape[0]
        )
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
    else:
        ax = display_3D_trajectory(
            trajectory,
            figsize=figure_size,
            one_shot=one_shot % trajectory.shape[0],
            per_plane=False,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()


def show_trajectory_full(trajectory, one_shot, figure_size, sample_freq=10):
    # General configuration
    nb_dim = trajectory.shape[-1]
    fig = plt.figure(figsize=(3 * figure_size, figure_size))
    subfigs = fig.subfigures(1, 3, wspace=0)

    # Trajectory display
    subfigs[0].suptitle("Trajectory", fontsize=displayConfig.fontsize, x=0.5, y=0.98)
    if nb_dim == 2:
        ax = display_2D_trajectory(
            trajectory,
            figsize=figure_size,
            one_shot=one_shot,
            subfigure=subfigs[0],
        )
    else:
        ax = display_3D_trajectory(
            trajectory,
            figsize=figure_size,
            one_shot=one_shot,
            per_plane=False,
            subfigure=subfigs[0],
        )
    for i in range(trajectory.shape[0]):
        ax.scatter(*(trajectory[i, ::sample_freq].T), s=15)
    ax.set_aspect("equal")

    # Gradient display
    subfigs[1].suptitle("Gradients", fontsize=displayConfig.fontsize, x=0.5, y=0.98)
    display_gradients_simply(
        trajectory,
        shot_ids=[one_shot],
        figsize=figure_size,
        subfigure=subfigs[1],
        uni_gradient="k",
        uni_signal="gray",
    )

    # Slew rates display
    subfigs[2].suptitle("Slew rates", fontsize=displayConfig.fontsize, x=0.5, y=0.98)
    display_gradients_simply(
        np.diff(trajectory, axis=1),
        shot_ids=[one_shot],
        figsize=figure_size,
        subfigure=subfigs[2],
        uni_gradient="k",
        uni_signal="gray",
    )

    subfigs[2].axes[0].set_ylabel("Sx")
    subfigs[2].axes[1].set_ylabel("Sy")
    if nb_dim == 2:
        subfigs[2].axes[2].set_ylabel("|S|")
    else:
        subfigs[2].axes[2].set_ylabel("Sz")
        subfigs[2].axes[3].set_ylabel("|S|")
    plt.show()


def show_trajectories(
    function, arguments, one_shot, subfig_size, dim="3D", axes=(0, 1)
):
    # Initialize trajectories with varying option
    trajectories = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    fig = plt.figure(
        figsize=(len(trajectories) * subfig_size, subfig_size),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(1, len(trajectories), wspace=0)
    for subfig, arg, traj in zip(subfigs, arguments, trajectories):
        if dim == "3D" and traj.shape[-1] == 3:
            ax = display_3D_trajectory(
                traj,
                figsize=subfig_size,
                one_shot=one_shot % traj.shape[0],
                subfigure=subfig,
                per_plane=False,
            )
        else:
            ax = display_2D_trajectory(
                traj[..., axes],
                figsize=subfig_size,
                one_shot=one_shot % traj.shape[0],
                subfigure=subfig,
            )
        labels = ["kx", "ky", "kz"]
        ax.set_xlabel(labels[axes[0]], fontsize=displayConfig.fontsize)
        ax.set_ylabel(labels[axes[1]], fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
    plt.show()


def show_density(density, figure_size, *, log_scale=False):
    density = density.T[::-1]

    plt.figure(figsize=(figure_size, figure_size))
    if log_scale:
        plt.imshow(density, cmap="jet", norm=colors.LogNorm())
    else:
        plt.imshow(density, cmap="jet")

    ax = plt.gca()
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    ax.set_aspect("equal")

    plt.axis(False)
    plt.colorbar()
    plt.show()


def show_densities(function, arguments, subfig_size, *, log_scale=False):
    # Initialize k-space densities with varying option
    densities = [function(arg).T[::-1] for arg in arguments]

    # Plot the trajectories side by side
    fig, axes = plt.subplots(
        1,
        len(densities),
        figsize=(len(densities) * subfig_size, subfig_size),
        constrained_layout=True,
    )

    for ax, arg, density in zip(axes, arguments, densities):
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
        ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
        ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        if log_scale:
            ax.imshow(density, cmap="jet", norm=colors.LogNorm())
        else:
            ax.imshow(density, cmap="jet")
        ax.axis(False)
    plt.show()


def show_locations(function, arguments, subfig_size, *, log_scale=False):
    # Initialize k-space locations with varying option
    locations = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    fig, axes = plt.subplots(
        1,
        len(locations),
        figsize=(len(locations) * subfig_size, subfig_size),
        constrained_layout=True,
    )

    for ax, arg, location in zip(axes, arguments, locations):
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
        ax.set_xlim(-1.05 * KMAX, 1.05 * KMAX)
        ax.set_ylim(-1.05 * KMAX, 1.05 * KMAX)
        ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
        ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        ax.scatter(location[..., 0], location[..., 1], s=3)
    plt.show()

# %% Import trajectory generation and processing functions
from mrinufft.trajectories.trajectory3D import initialize_3D_phyllotaxis_radial
from mrinufft.trajectories.trajectory3D import initialize_3D_wave_caipi
from mrinufft.io.nsp import write_trajectory
from mrinufft.trajectories.trajectory3D import initialize_3D_cones
from mrinufft.trajectories.projection import parameterize_by_arc_length
from mrinufft.trajectories.trajectory3D import stack


# %% Define Trajectory and acquisition parameters

acq_na = Acquisition(
    fov=(0.180, 0.180, 0.180),
    img_size=(60, 60, 60),
    gamma=Gammas.SODIUM,
)

# Trajectory parameters
Nc = 40 # Number of shots
Ns = 100  # Number of samples per shot
# in_out = True # Choose between in-out or center-out trajectories
tilt = "uniform"  # Angular distance between shots
nb_repetitions = 30  # Number of stacks, rotations, cones, shells etc.
nb_revolutions = 3  # Number of revolutions for base trajectories
seed = 0  # Seed for random trajectories

# Display parameters
figure_size = 10  # Figure size for trajectory plots
subfigure_size = 6  # Figure size for subplots
one_shot = -5  # Highlight one shot in particular
# %%
from mrinufft.trajectories.trajectory3D import initialize_3D_golden_means_radial
from mrinufft.trajectories.trajectory3D import initialize_3D_wave_caipi
from mrinufft.io.nsp import write_trajectory
from mrinufft.trajectories.trajectory3D import initialize_3D_cones
from mrinufft.trajectories.projection import parameterize_by_arc_length
from mrinufft.trajectories.tools import add_slew_ramp
from mrinufft.trajectories.trajectory3D import stack
from mrinufft.trajectories.tools import add_slew_ramp_to_traj_func
# %% STEP 1 — Generatate and display initial trajectory

# trajectory = initialize_3D_golden_means_radial(Nc, Ns, in_out=False)
# trajectory = mn.initialize_3D_wave_caipi(Nc, Ns)

from mrinufft.trajectories import conify, initialize_2D_spiral, initialize_2D_rosette, initialize_2D_fibonacci_spiral
# traj2D=initialize_2D_rosette(Nc , Ns,coprime_index=1,in_out=False)
# traj2D=initialize_2D_spiral(Nc , Ns,in_out=False)

# #trajectory=stack(traj2D, nb_stacks=nb_repetitions )
# trajectory = conify(traj2D, nb_cones=12,in_out=False)

trajectory = initialize_3D_phyllotaxis_radial(Nc, Ns, in_out=False)


show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)
show_trajectory_full(trajectory, one_shot=one_shot % Nc, figure_size=6, sample_freq=10)
grads, slews = compute_gradients_and_slew_rates(trajectory,acq=acq_na)
grad_max = float(np.max(np.abs(grads)))
slew_max = float(np.max(np.abs(slews)))
print(f"Before Optimization and constraint management :  Max gradient: {grad_max:.6f} T/m, Max slew rate: {slew_max:.6f} T/m/ms")
# %% Optionnal 2D Trajectory Sanity Check
show_trajectory(traj2D, figure_size=figure_size, one_shot=one_shot)
show_trajectory_full(traj2D, one_shot=one_shot, figure_size=6, sample_freq=10)
grads, slews = compute_gradients_and_slew_rates(traj2D,acq=acq_na)
grad_max = float(np.max(np.abs(grads)))
slew_max = float(np.max(np.abs(slews)))
print(f"2D sub-trajectory : Max gradient: {grad_max:.6f} T/m, Max slew rate: {slew_max:.6f} T/m/ms")

# %% STEP 2 — Arc-length parameterization + display 
trajectory = parameterize_by_arc_length(trajectory)
show_trajectory_full(trajectory, one_shot=one_shot % Nc, figure_size=6, sample_freq=10)
grads, slews = compute_gradients_and_slew_rates(trajectory,acq=acq_na)
grad_max = float(np.max(np.abs(grads)))
slew_max = float(np.max(np.abs(slews)))
print(f"POST ARC LENGTH : Max gradient: {grad_max:.6f} T/m, Max slew rate: {slew_max:.6f} T/m/ms")

# %% STEP 3 - Slew Rate Management (prephasor + spoiler, "lp" | "lp-minslew" | "osqp" )

from mrinufft.trajectories.utils import (
    convert_trajectory_to_gradients,
    convert_gradients_to_trajectory,
)
from mrinufft.trajectories.gradients import get_prephasors_and_spoilers

traj_grad, init_points = convert_trajectory_to_gradients(trajectory, acq=acq_na)

method = "lp-minslew"  # "lp" | "lp-minslew" | "osqp"
p, s = get_prephasors_and_spoilers(
    trajectory,
    acq=acq_na,
    method=method,
    spoil_loc=(0, 0, 1),
)

g_full = np.concatenate([p, traj_grad, s], axis=1)
trajectory_full = convert_gradients_to_trajectory(
    g_full,
    np.zeros_like(init_points),
    acq=acq_na,
)
# trajectory=trajectory_full

# %% CONNECT SHOTS (ramp-up/down) — replace get_prephasors_and_spoilers()
# Goal: only connect end-of-shot i -> start-of-shot i+1 under (gmax, smax) constraints,
#       without forcing a "spoil_loc" or "prephase_loc".

import numpy as np
from mrinufft.trajectories.utils import (
    convert_trajectory_to_gradients,
    convert_gradients_to_trajectory,
)
from mrinufft.trajectories.gradients import connect_gradient, min_length_connection


def connect_shots_only(
    trajectory,                   # (Nshots, Ns, 3) normalized k-space trajectory
    acq,
    method="lp-minslew",          # "lp" | "lp-minslew" | "osqp"
    Nconn=None,                   # if None: auto minimal length that works for all connections
):
    """
    Returns:
        trajectory_connected : (Nshots, Ns-1 + Nconn, 3) normalized k-space traj
        Nconn               : int, connection length actually used
    """

    # 1) k-space traj -> gradients + start/end k-points (k in m^-1)
    grads, kstarts, kends = convert_trajectory_to_gradients(
        trajectory, acq=acq, get_final_positions=True
    )
    # grads: (Nshots, Ns-1, 3)
    gstarts = grads[:, 0, :]     # (Nshots, 3)
    gends   = grads[:, -1, :]    # (Nshots, 3)

    nshots = trajectory.shape[0]
    if nshots < 2:
        raise ValueError("Need at least 2 shots to connect.")

    # 2) Build arrays for connections: end(i) -> start(i+1)
    k_from = kends[:-1, :]       # (Nshots-1, 3)
    k_to   = kstarts[1:, :]      # (Nshots-1, 3)
    g_from = gends[:-1, :]       # (Nshots-1, 3)
    g_to   = gstarts[1:, :]      # (Nshots-1, 3)

    # 3) Choose minimal connection length if not provided
    if Nconn is None:
        Nconn = min_length_connection(k_from, k_to, g_from, g_to, acq=acq, method=method)

    # 4) Solve all connections in gradient domain
    conn = connect_gradient(
        kstarts=k_from,
        kends=k_to,
        gstarts=g_from,
        gends=g_to,
        acq=acq,
        method=method,
        N=Nconn,
    )
    # conn: (Nshots-1, Nconn, 3)

    # 5) Stitch per-shot gradient blocks: [shot_grad] + [connection_to_next]
    #    (keep same "Nshots blocks" structure — easy to keep using your plotting/export)
    Ns_grad = grads.shape[1]  # = Ns-1
    g_full = np.zeros((nshots, Ns_grad + Nconn, 3), dtype=grads.dtype)

    for i in range(nshots):
        g_full[i, :Ns_grad, :] = grads[i]
        if i < nshots - 1:
            g_full[i, Ns_grad:, :] = conn[i]
        else:
            # last shot: no "next shot" -> keep flat (0) tail (or replace by a ramp-to-zero if you WANT)
            g_full[i, Ns_grad:, :] = 0.0

    # 6) Integrate gradients back to a normalized k-space trajectory
    # Initial positions are the true kstarts of each shot (in m^-1)
    trajectory_connected = convert_gradients_to_trajectory(
        g_full,
        initial_positions=kstarts,   # IMPORTANT: keep shot-specific k starts
        acq=acq,
    )

    return trajectory_connected, Nconn


# -------------------------
# USE IT (drop-in replacement)
# -------------------------
# BEFORE (remove this):
# p, s = get_prephasors_and_spoilers(...)
# g_full = np.concatenate([p, traj_grad, s], axis=1)
# trajectory_full = convert_gradients_to_trajectory(...)

# AFTER (use this):
trajectory, Nconn = connect_shots_only(
    trajectory,
    acq=acq_na,
    method="lp-minslew",   # or "osqp"
    Nconn=200,            # auto minimal length
)
print(f"[INFO] Used Nconn = {Nconn} samples for each inter-shot connection.")






#%% ============================================================
# Block 1 — Function: REPLACE edges + (optionally) ADD samples in ramps
# Uses: from mrinufft.trajectories.gradients import connect_gradient
# ============================================================

import numpy as np
from mrinufft.trajectories.utils import (
    unnormalize_trajectory,
    convert_trajectory_to_gradients,
    convert_gradients_to_trajectory,
)
from mrinufft.trajectories.gradients import connect_gradient


def add_edge_ramps_with_connect_gradient(
    trajectory_norm: np.ndarray,   # (Nshots, Ns, 3) NUFFT-normalized k
    acq,
    # how many original points you WANT to remove/replace at edges
    n_replace_start: int = 0,
    n_replace_end: int = 0,
    # ramp length in k-points:
    #   - None  => auto minimal feasible (solver chooses N, so k-points = N+1)
    #   - int>=2 => force that many k-points (so N = n_ramp-1)
    n_ramp_start: int | None = None,
    n_ramp_end: int | None = None,
    method: str = "osqp",  # "lp" | "lp-minslew" | "osqp"
    # boundary gradients you want at very start/end (recommended: 0)
    g_start_target=(0.0, 0.0, 0.0),
    g_end_target=(0.0, 0.0, 0.0),
    # behavior when ramp produces MORE points than needed:
    #   - "keep_all": keep full ramp => Ns increases
    #   - "match_replace": crop ramp so exactly replaces removed points => Ns stays ~same
    start_policy: str = "keep_all",
    end_policy: str = "keep_all",
):
    """
    Build clean ramps at beginning/end using connect_gradient and INSERT them.

    - We remove n_replace_start points at the beginning and n_replace_end at the end.
    - Then we add a ramp-up and/or ramp-down.
    - If ramps are longer than the removed windows, total Ns increases automatically.

    Indexing convention:
      interior = original k[ n_replace_start : Ns - n_replace_end ]

      start ramp connects:
        k0 (original first kept point)???  -> actually we connect from original k[0]
        to interior start point k[i_start], matching gradient leaving that point.

      end ramp connects:
        from interior end point k[i_end] to original last point k[-1],
        matching gradient entering/leaving boundaries.

    Practical meaning:
    - You can set n_replace_* small (e.g. 2–10) to remove bad discontinuities.
    - Let n_ramp_* = None so the solver chooses minimal feasible ramps.
    - With start_policy/end_policy="keep_all", if the solver needs more points, you KEEP them.
    """
    traj = np.asarray(trajectory_norm)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError("trajectory_norm must have shape (Nshots, Ns, 3)")
    nshots, Ns, ndim = traj.shape

    if n_replace_start < 0 or n_replace_end < 0:
        raise ValueError("n_replace_start/end must be >= 0")
    if n_replace_start + n_replace_end >= Ns:
        raise ValueError("n_replace_start + n_replace_end must be < Ns")

    if start_policy not in ("keep_all", "match_replace"):
        raise ValueError("start_policy must be 'keep_all' or 'match_replace'")
    if end_policy not in ("keep_all", "match_replace"):
        raise ValueError("end_policy must be 'keep_all' or 'match_replace'")

    # Interior indices in original trajectory
    i_start = n_replace_start
    i_end_excl = Ns - n_replace_end
    if i_end_excl - i_start < 2:
        raise ValueError("Interior too small after replacement. Reduce n_replace_*.")

    # Convert normalized k -> physical k (m^-1) because connect_gradient expects unnormalized k
    k_m1 = unnormalize_trajectory(traj, acq)  # (Nshots, Ns, 3)

    # Gradients of the original readout (for boundary gradient matching)
    g_read, kstarts_m1, kends_m1 = convert_trajectory_to_gradients(traj, acq, get_final_positions=True)
    # g_read: (Nshots, Ns-1, 3)

    # Interior k block (kept as-is)
    interior_norm = traj[:, i_start:i_end_excl, :]  # normalized k

    # Interior boundary points in m^-1
    k_start_interior = k_m1[:, i_start, :]
    k_end_interior = k_m1[:, i_end_excl - 1, :]

    # Boundary gradients at interior boundaries (from original readout)
    # Gradient leaving interior start:
    g_start_interior = g_read[:, i_start, :]  # step k[i_start]->k[i_start+1]
    # Gradient entering interior end approximated by last kept gradient:
    g_end_interior = g_read[:, i_end_excl - 2, :]  # step k[i_end_excl-2]->k[i_end_excl-1]

    # Global endpoints in m^-1 (original)
    k0 = k_m1[:, 0, :]
    k_last = k_m1[:, -1, :]

    g_start_target = np.tile(np.array(g_start_target, float)[None, :], (nshots, 1))
    g_end_target = np.tile(np.array(g_end_target, float)[None, :], (nshots, 1))

    # --------------------
    # START RAMP
    # --------------------
    ramp_start_norm = np.zeros((nshots, 0, 3), dtype=traj.dtype)
    ramp_start_pts = 0
    if n_replace_start > 0:
        if n_ramp_start is None:
            N_start = None
        else:
            if n_ramp_start < 2:
                raise ValueError("n_ramp_start must be >=2 (k-points) or None")
            N_start = n_ramp_start - 1

        g_ramp_start = connect_gradient(
            kstarts=k0,
            kends=k_start_interior,
            gstarts=g_start_target,
            gends=g_start_interior,
            acq=acq,
            method=method,
            N=N_start,  # None => auto
        )
        traj_ramp_start_norm = convert_gradients_to_trajectory(
            g_ramp_start, initial_positions=k0, acq=acq
        )  # normalized, shape (Nshots, N+1, 3)

        # We must avoid duplicating k_start_interior (already in interior_norm first sample)
        # So we drop the last ramp point.
        full = traj_ramp_start_norm[:, :-1, :]
        ramp_start_pts = full.shape[1]

        if start_policy == "match_replace":
            # keep exactly n_replace_start points (replace window length)
            if ramp_start_pts < n_replace_start:
                raise RuntimeError(
                    f"Auto ramp start has {ramp_start_pts} pts < n_replace_start={n_replace_start}. "
                    f"Set n_ramp_start to >= {n_replace_start+1} or use start_policy='keep_all'."
                )
            ramp_start_norm = full[:, -n_replace_start:, :]
            ramp_start_pts = ramp_start_norm.shape[1]
        else:
            # keep all ramp points => can increase Ns
            ramp_start_norm = full

    # --------------------
    # END RAMP
    # --------------------
    ramp_end_norm = np.zeros((nshots, 0, 3), dtype=traj.dtype)
    ramp_end_pts = 0
    if n_replace_end > 0:
        if n_ramp_end is None:
            N_end = None
        else:
            if n_ramp_end < 2:
                raise ValueError("n_ramp_end must be >=2 (k-points) or None")
            N_end = n_ramp_end - 1

        g_ramp_end = connect_gradient(
            kstarts=k_end_interior,
            kends=k_last,
            gstarts=g_end_interior,
            gends=g_end_target,
            acq=acq,
            method=method,
            N=N_end,  # None => auto
        )
        traj_ramp_end_norm = convert_gradients_to_trajectory(
            g_ramp_end, initial_positions=k_end_interior, acq=acq
        )  # normalized, shape (Nshots, N+1, 3)

        # Avoid duplicating k_end_interior (already last sample of interior_norm)
        # So we drop the first ramp point.
        full = traj_ramp_end_norm[:, 1:, :]
        ramp_end_pts = full.shape[1]

        if end_policy == "match_replace":
            if ramp_end_pts < n_replace_end:
                raise RuntimeError(
                    f"Auto ramp end has {ramp_end_pts} pts < n_replace_end={n_replace_end}. "
                    f"Set n_ramp_end to >= {n_replace_end+1} or use end_policy='keep_all'."
                )
            ramp_end_norm = full[:, :n_replace_end, :]
            ramp_end_pts = ramp_end_norm.shape[1]
        else:
            ramp_end_norm = full

    # Assemble: ramp_start + interior + ramp_end
    traj_out = np.concatenate([ramp_start_norm, interior_norm, ramp_end_norm], axis=1)

    info = {
        "Ns_in": int(Ns),
        "Ns_out": int(traj_out.shape[1]),
        "added_pts_start": int(max(0, ramp_start_pts - n_replace_start)) if start_policy == "keep_all" else 0,
        "added_pts_end": int(max(0, ramp_end_pts - n_replace_end)) if end_policy == "keep_all" else 0,
        "ramp_start_pts": int(ramp_start_pts),
        "ramp_end_pts": int(ramp_end_pts),
        "n_replace_start": int(n_replace_start),
        "n_replace_end": int(n_replace_end),
        "n_ramp_start": None if n_ramp_start is None else int(n_ramp_start),
        "n_ramp_end": None if n_ramp_end is None else int(n_ramp_end),
        "method": method,
        "start_policy": start_policy,
        "end_policy": end_policy,
    }
    return traj_out, info

#%% ============================================================
# Block 2 — Use + verify BEFORE/AFTER with prints (first/last points, |G| and |S|)
# ============================================================

import numpy as np
from mrinufft.trajectories.utils import (
    unnormalize_trajectory,
    convert_trajectory_to_gradients,
)

# Use your current trajectory (normalized k-space)
traj_before = trajectory.copy()
shot_id = 0
n_print = 10

print("\n=== BEFORE: first k (normalized) ===")
print(traj_before[shot_id, :n_print, :])
print("\n=== BEFORE: last k (normalized) ===")
print(traj_before[shot_id, -n_print:, :])

# Grad/slew BEFORE (explicit units)
gB, _ = convert_trajectory_to_gradients(traj_before, acq=acq_na)  # T/m
sB = np.diff(gB, axis=1) / acq_na.raster_time                    # T/m/s
print("\n[BEFORE] max|G| =", float(np.max(np.linalg.norm(gB, axis=-1))), "T/m")
print("[BEFORE] max|S| =", float(np.max(np.linalg.norm(sB, axis=-1))), "T/m/s")

# --- Apply: for example remove 5 points at start and 5 at end, but KEEP ALL ramp points if solver needs more
traj_after, info = add_edge_ramps_with_connect_gradient(
    trajectory_norm=traj_before,
    acq=acq_na,
    n_replace_start=10,
    n_replace_end=20,
    n_ramp_start=None,     # AUTO minimal feasible (can be > n_replace_start+1)
    n_ramp_end=None,       # AUTO minimal feasible
    method="lp-minslew",  # "lp" | "lp-minslew" | "osqp"
    start_policy="keep_all",  
    end_policy="keep_all",
    g_start_target=(0, 0, 0),
    g_end_target=(0, 0, 0),
)

print("\n[INFO]", info)
print(f"[INFO] trajectory.shape: {traj_before.shape} -> {traj_after.shape}")

print("\n=== AFTER: first k (normalized) ===")
print(traj_after[shot_id, :n_print, :])
print("\n=== AFTER: last k (normalized) ===")
print(traj_after[shot_id, -n_print:, :])

# Grad/slew AFTER
gA, _ = convert_trajectory_to_gradients(traj_after, acq=acq_na)   # T/m
sA = np.diff(gA, axis=1) / acq_na.raster_time                    # T/m/s
print("\n[AFTER]  max|G| =", float(np.max(np.linalg.norm(gA, axis=-1))), "T/m")
print("[AFTER]  max|S| =", float(np.max(np.linalg.norm(sA, axis=-1))), "T/m/s")

Ns= traj_after.shape[1]
print(Ns)
# Use the new trajectory
trajectory = traj_after

# %%# Show the results

show_trajectory_full(trajectory, one_shot=one_shot % Nc, figure_size=6, sample_freq=10)

grads, slews = compute_gradients_and_slew_rates(trajectory, acq=acq_na)
grad_max = float(np.max(np.abs(grads)))
slew_max = float(np.max(np.abs(slews)))
print(f"POST Ramp Connection : Max gradient: {grad_max:.6f} T/m, Max slew rate: {slew_max:.6f} T/m/ms")

#%% STEP 2.5 — Optionnal Center anomaly CORRECTION + display
from mrinufft.trajectories.gradients import patch_center_anomaly

# -------------------------------------------------------------------------
# Optional: patch center anomaly (per-shot) BEFORE prephaser/spoiler
# -------------------------------------------------------------------------
# patch_center_anomaly works on a single shot of shape (Ns, Nd).
# Apply it to every shot independently.
# Set in_out=True for in-out shots, False for center-out shots.
# -------------------------------------------------------------------------
do_patch = True
patch_in_out = True   # True if your shot is in-out; False if center-out
lr = 1e-1             # learning rate for the local optimization

if do_patch:
    traj_patched = np.empty_like(trajectory)
    for i in range(trajectory.shape[0]):
        traj_patched[i], _ = patch_center_anomaly(
            trajectory[i],
            in_out=patch_in_out,
            learning_rate=lr,
        )
    trajectory = traj_patched
# %% STEP 4 — Display after Optimization
show_trajectory_full(trajectory, one_shot=one_shot % Nc, figure_size=6, sample_freq=10)
grads, slews = compute_gradients_and_slew_rates(trajectory,acq=acq_na)
grad_max = float(np.max(np.abs(grads)))
slew_max = float(np.max(np.abs(slews)))
print(f"POST Constraint managment : Max gradient: {grad_max:.6f} T/m, Max slew rate: {slew_max:.6f} T/m/ms")

# %% STEP 2 — Optionnal novel Arc-length parameterization + display 
trajectory = parameterize_by_arc_length(trajectory)
show_trajectory_full(trajectory, one_shot=one_shot % Nc, figure_size=6, sample_freq=10)
grads, slews = compute_gradients_and_slew_rates(trajectory,acq=acq_na)
grad_max = float(np.max(np.abs(grads)))
slew_max = float(np.max(np.abs(slews)))
print(f"SECOND ARC LENGTH : Max gradient: {grad_max:.6f} T/m, Max slew rate: {slew_max:.6f} T/m/ms")

# %% EXPORT BINARY
write_trajectory(
    trajectory=trajectory,
    FOV=(0.180, 0.180, 0.180),
    img_size=(60, 60, 60), # 3 mm iso
    # img_size=(38.4, 38.4, 28.2), # 5 mm iso
    grad_filename=f"Rosette_Conify_Ncone{nb_repetitions}_constr_CONNECT_SODIUM_Nc{Nc}_Ns{Ns}",
    gamma=11.262e3,
    smax=0.2, #0.16
    gmax=0.04,
    # pregrad="prephase",
    # postgrad="lp-minslew",
    check_constraints=True,
)
# %%
