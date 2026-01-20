from typing import NamedTuple

import matplotlib.pyplot as plt

from syssim.core import Node, InputPort

class NodeScopeInputs(NamedTuple):

    scope: InputPort
    """The input vector for the scope"""

class NodeScope(Node):
    def __init__(self, **kwargs):
        """Time-series plotting node using matplotlib.

        Configuration (TOML)
        --------------------
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

        self._i = NodeScopeInputs(InputPort("input_scope", self))
        super().__init__(self._i, None, **kwargs)

    def initialize(self):
        self._title = self._config.get("title", "Scope")
        self._xlabel = self._config.get("xlabel", "Time")
        self._ylabel = self._config.get("ylabel", "Value")
        self._legend = self._config.get("legend", [])
        self._show = self._config.get("show", True)
        self._save = self._config.get("save", False)

    def update(self, sim_time: float):
        self._t.append(sim_time)
        self._y.append(self._i.scope.read())

    def finalize(self, fault_history = None):
        plt.figure
        plt.plot(self._t, self._y)
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title(self._title)
        plt.grid(True)
        plt.legend(self._legend)

        if self._save:
            if self._system.get_output_dir() is not None:
                plt.savefig(
                    self._system.get_output_dir() + f"{self.name}.png".replace(" ", "_")
                )
            else:
                print(
                    f"Cannot save figure from node {self.name} because no output dir was provided to the simulation."
                )
        if self._show:
            plt.show()

    @property
    def i(self):
        return self._i
    
    @property
    def o(self):
        return ()
