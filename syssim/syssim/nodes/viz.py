from typing import NamedTuple

import matplotlib.pyplot as plt

from syssim.core import Node, InputPort

class NodeScopeInputs(NamedTuple):

    scope: InputPort
    """The input vector for the scope"""

class NodeScope(Node):
    def __init__(self, **kwargs):
        """Node implementing a scope. Displays values recorded in the simulation in a matplotlib plot.

        Ports:
            input_scope (np.array): input for the values to be recorded and plotted

        Configs:
            title (str): graph title
            xlabel (str): graph x label
            ylabal (str): graph y label
            save (bool): should we save the figure?
            show (bool): should we show the figure?
        """
        self._t = list()
        self._y = list()

        self._i = NodeScopeInputs(InputPort("input_scope", self))
        super().__init__(self._i, None, **kwargs)

    def initialize(self):
        self._title = self._config["title"]
        self._xlabel = self._config["xlabel"]
        self._ylabel = self._config["ylabel"]
        self._show = self._config["show"]
        self._save = self._config["save"]

    def update(self, sim_time: float):
        self._t.append(sim_time)
        self._y.append(self._i.scope.read())

    def finalize(self):
        plt.figure
        plt.plot(self._t, self._y)
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title(self._title)
        plt.grid(True)

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
