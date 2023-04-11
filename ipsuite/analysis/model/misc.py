import matplotlib.pyplot as plt
import zntrack


class MultiXYPlot(zntrack.Node):
    """Plot multiple BoxScale energy curces on the same plot."""

    nodes: list = zntrack.zn.deps()
    output_file: str = zntrack.dvc.outs(zntrack.nwd / "plot.png")

    def run(self):
        fig, ax = plt.subplots()
        for node in self.nodes:
            x = node.energies["x"]
            y = node.energies["y"]
            ax.plot(x, y, label=node.name)

        ax.legend()
        fig.savefig(self.output_file)
