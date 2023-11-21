from typing import List, Tuple
from matplotlib.axes import Axes
from matplotlib.artist import Artist


def add_fault_vbars(faults: List, ax: Axes, alpha=0.5, time_scale=3600) -> List[Artist]:
    output = []
    for bf in faults:
        if bf.is_occuring():
            output.append(
                ax.axvspan(
                    bf.start_time() / time_scale,
                    (bf.start_time() + bf.duration()) / time_scale,
                    label=f"basic-fault.{bf.get_name()}",
                    alpha=alpha,
                )
            )
    return output


def add_fault_detect_vline(
    faults: List[Tuple], ax: Axes, time_scale=3600
) -> List[Artist]:
    output = []
    for f in faults:
        output.append(
            ax.axvline(f[1] / time_scale, color="r", ls="-.", label="fault detection")
        )
        ybottom, ytop = ax.get_ylim()
        ax.text(f[1] / time_scale, (ytop + ybottom) / 2, f[0], fontsize=10)
    return output
