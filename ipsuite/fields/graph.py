"""Lazy ASE Atoms loading."""
import json
import pathlib
import typing

import networkx as nx
import zntrack


class NxGraph(zntrack.Field):
    """Store list[ase.Atoms] in an ASE database."""

    dvc_option = "--outs"
    group = zntrack.FieldGroup.RESULT

    def __init__(self):
        super().__init__(use_repr=False)

    def get_files(self, instance: zntrack.Node) -> list:
        return [(instance.nwd / f"{self.name}.json").as_posix()]

    def get_stage_add_argument(self, instance: zntrack.Node) -> typing.List[tuple]:
        return [(self.dvc_option, file) for file in self.get_files(instance)]

    def save(self, instance: zntrack.Node):
        """Save value with ase.db.connect."""
        try:
            graph: nx.Graph = getattr(instance, self.name)
        except AttributeError:
            return
        instance.nwd.mkdir(exist_ok=True, parents=True)
        file = self.get_files(instance)[0]
        with pathlib.Path(file).open("w") as f:
            json.dump(graph, f, default=nx.node_link_data)

    def get_data(self, instance: zntrack.Node) -> typing.List[nx.Graph]:
        """Get graph File."""
        file = self.get_files(instance)[0]
        with pathlib.Path(file).open("r") as f:
            return [nx.node_link_graph(x) for x in json.load(f)]
