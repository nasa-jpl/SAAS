"""CLI entrypoint for asteroid flyby simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import tyro

from .flyby_sim import FlybyRunConfig, run_flyby


@dataclass
class CliArgs:
    """Run a syssim asteroid flyby with reaction-wheel pointing control."""

    run: FlybyRunConfig = field(default_factory=FlybyRunConfig)


def main() -> int:
    """Run the configured flyby simulation."""
    args = tyro.cli(CliArgs)
    artifacts = run_flyby(args.run)

    print(f"Output directory: {artifacts.output_dir}")
    print(f"Trajectory animation: {artifacts.trajectory_animation}")
    print(f"Timeseries CSV: {artifacts.log_csv}")
    print(f"Timeseries NPZ: {artifacts.log_npz}")
    if artifacts.rendered_video is not None:
        print(f"Rendered video: {artifacts.rendered_video}")
    else:
        print("Rendered video: disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
