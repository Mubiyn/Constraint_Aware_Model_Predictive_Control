from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _span_where(mask: np.ndarray) -> tuple[float, float] | None:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return float(idx[0]), float(idx[-1])


def plot_run(npz_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    data = _load_npz(npz_path)

    t = data.get("t")
    zmp = data.get("zmp")
    zmp_bounds = data.get("zmp_bounds")
    zmp_violation = data.get("zmp_violation")
    dcm = data.get("dcm")
    dcm_ref = data.get("dcm_ref")
    dcm_err = data.get("dcm_err")
    com_vel = data.get("com_vel")
    v_ref = data.get("v_ref")
    solve_ms = data.get("solve_ms")
    push = data.get("push")
    stride = data.get("stride")
    t_ss = data.get("t_ss")
    t_ds = data.get("t_ds")
    margin = data.get("margin")

    artifacts: list[Path] = []

    def add_push_span(ax):
        if t is None or push is None:
            return
        push_mask = np.asarray(push) > 0.5
        if not np.any(push_mask):
            return
        span = _span_where(push_mask)
        if span is None:
            return
        i0, i1 = int(span[0]), int(span[1])
        ax.axvspan(float(t[i0]), float(t[i1]), alpha=0.15, color="red", label="push")

    # ZMP vs bounds
    if t is not None and zmp is not None and zmp_bounds is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        finite_bounds = np.isfinite(zmp_bounds).all(axis=1)

        axes[0].plot(t, zmp[:, 0], label="zmp_x")
        axes[0].plot(t[finite_bounds], zmp_bounds[finite_bounds, 0], "k--", linewidth=1, label="xmin")
        axes[0].plot(t[finite_bounds], zmp_bounds[finite_bounds, 1], "k--", linewidth=1, label="xmax")
        add_push_span(axes[0])
        axes[0].set_ylabel("x (m)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, zmp[:, 1], label="zmp_y")
        axes[1].plot(t[finite_bounds], zmp_bounds[finite_bounds, 2], "k--", linewidth=1, label="ymin")
        axes[1].plot(t[finite_bounds], zmp_bounds[finite_bounds, 3], "k--", linewidth=1, label="ymax")
        add_push_span(axes[1])
        axes[1].set_ylabel("y (m)")
        axes[1].set_xlabel("time (s)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        if zmp_violation is not None and np.any(np.isfinite(zmp_violation)):
            viol = np.asarray(zmp_violation)
            title_extra = f" | max violation={np.nanmax(viol):.4f} m"
        else:
            title_extra = ""

        fig.suptitle(f"{npz_path.stem}: ZMP vs bounds{title_extra}")
        fig.tight_layout()
        out = out_dir / f"{npz_path.stem}_zmp_bounds.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        artifacts.append(out)

    # DCM tracking
    if t is not None and dcm is not None and dcm_ref is not None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, dcm[:, 0], label="dcm_x")
        axes[0].plot(t, dcm_ref[:, 0], label="dcm_ref_x")
        add_push_span(axes[0])
        axes[0].set_ylabel("x (m)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, dcm[:, 1], label="dcm_y")
        axes[1].plot(t, dcm_ref[:, 1], label="dcm_ref_y")
        add_push_span(axes[1])
        axes[1].set_ylabel("y (m)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        if dcm_err is not None:
            axes[2].plot(t, dcm_err, label="dcm_err")
        else:
            err = np.linalg.norm(dcm - dcm_ref, axis=1)
            axes[2].plot(t, err, label="dcm_err")
        add_push_span(axes[2])
        axes[2].set_ylabel("err (m)")
        axes[2].set_xlabel("time (s)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f"{npz_path.stem}: DCM tracking")
        fig.tight_layout()
        out = out_dir / f"{npz_path.stem}_dcm.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        artifacts.append(out)

    # Velocity tracking
    if t is not None and com_vel is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, com_vel[:, 0], label="com_vel_x")
        if v_ref is not None and np.any(np.isfinite(v_ref)):
            axes[0].plot(t, v_ref, label="v_ref", linewidth=1.5)
        add_push_span(axes[0])
        axes[0].set_ylabel("m/s")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        if stride is not None and t_ss is not None and t_ds is not None and margin is not None:
            axes[1].plot(t, stride, label="stride (m)")
            axes[1].plot(t, t_ss, label="t_ss (s)")
            axes[1].plot(t, t_ds, label="t_ds (s)")
            axes[1].plot(t, margin, label="zmp_margin (m)")
            axes[1].legend(loc="upper right", ncol=2)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlabel("time (s)")

        fig.suptitle(f"{npz_path.stem}: velocity + gait params")
        fig.tight_layout()
        out = out_dir / f"{npz_path.stem}_velocity.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        artifacts.append(out)

    # Solve time
    if t is not None and solve_ms is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(t, solve_ms, label="solve_ms")
        add_push_span(ax)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ms")
        ax.grid(True, alpha=0.3)
        if np.any(np.isfinite(solve_ms)):
            ax.set_title(f"{npz_path.stem}: QP solve time (max={np.nanmax(solve_ms):.2f} ms)")
        else:
            ax.set_title(f"{npz_path.stem}: QP solve time")
        fig.tight_layout()
        out = out_dir / f"{npz_path.stem}_solve_ms.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        artifacts.append(out)

    return artifacts


def plot_directory(input_dir: Path, pattern: str = "*.npz") -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    out_dir = input_dir / "plots"

    artifacts: list[Path] = []
    for npz_path in sorted(input_dir.glob(pattern)):
        artifacts.extend(plot_run(npz_path, out_dir))
    return artifacts


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot ATP results from saved .npz logs")
    parser.add_argument("--in", dest="input_dir", type=str, required=True, help="Directory with .npz logs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    plot_directory(Path(args.input_dir))


if __name__ == "__main__":
    main()
