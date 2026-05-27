from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from syssim.core import EmptySpec, InputPort, Node, input_port


@dataclass
class NodeScopeInputs:
    """Input specification for ``NodeScope``.

    Attributes
    ----------
    scope : InputPort
        Signal value to record and plot.
    """

    scope: InputPort[object] = input_port(object)
    """Input port receiving the signal value to record and plot."""


@dataclass
class NodeScopeConfig:
    """Configuration for ``NodeScope`` plots.

    Attributes
    ----------
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    legend : list of str
        Legend entries passed to ``plt.legend``.
    show : bool
        Display the figure interactively after simulation finalizes.
    save : bool
        Save the figure to the active simulation output directory.
    """

    title: str = "Scope"
    """Plot title."""
    xlabel: str = "Time"
    """X-axis label."""
    ylabel: str = "Value"
    """Y-axis label."""
    legend: list[str] = field(default_factory=list)
    """Legend entries passed to ``plt.legend``."""
    show: bool = True
    """Display the figure interactively after the simulation finalizes."""
    save: bool = False
    """Save the figure to the simulation output directory as ``<node_name>.png``."""


class NodeScope(Node[NodeScopeInputs, EmptySpec, EmptySpec, NodeScopeConfig]):
    """Record an input signal over time and plot it with matplotlib.

    Parameters
    ----------
    **kwargs
        Arguments forwarded to ``Node``. Pass ``config=NodeScopeConfig(...)``
        to customize the plot.
    """

    Inputs = NodeScopeInputs
    Config = NodeScopeConfig

    def __init__(self, **kwargs):
        """Time-series plotting node using matplotlib.

        Configuration
        -------------
        Pass a :class:`NodeScopeConfig` dataclass instance as ``config``.

        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        legend : list, optional
            Legend entries passed to ``plt.legend``.
        save : bool, optional
            Save figure under the simulation output directory.
        show : bool, optional
            Display the plot interactively.
        """
        self._t = list()
        self._y = list()
        super().__init__(**kwargs)

    def initialize(self):
        """Load plotting configuration before a run."""
        self._title = self.config.title
        self._xlabel = self.config.xlabel
        self._ylabel = self.config.ylabel
        self._legend = self.config.legend
        self._show = self.config.show
        self._save = self.config.save

    def update(self, sim_time: float):
        """Record the current input sample value.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.
        """
        self._t.append(sim_time)
        self._y.append(self.i.scope.read().value)

    def finalize(self, fault_history = None):
        """Create, show, and optionally save the recorded plot.

        Parameters
        ----------
        fault_history : dict, optional
            Mapping from simulation time to fault active states. Accepted for
            compatibility with the node lifecycle.
        """
        plt.figure()
        plt.plot(self._t, self._y)
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title(self._title)
        plt.grid(True)
        plt.legend(self._legend)

        if self._save:
            if self._system.get_output_dir() is not None:
                plt.savefig(
                    f"{self._system.get_output_dir()}/{self.name}.png".replace(" ", "_")
                )
            else:
                print(
                    f"Cannot save figure from node {self.name} because no output dir was provided to the simulation."
                )
        if self._show:
            plt.show()
