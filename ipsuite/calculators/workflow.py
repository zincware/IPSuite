import quippy.potential
import zntrack
from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec

from ipsuite import base, utils
from ipsuite.models import GAP


class Workflow(base.ProcessAtoms):
    """Parallel Execution of ASE calculators using the WFL package."""
    calculator_node: zntrack.Node = zntrack.zn.deps()

    properties = zntrack.zn.params(["energy", "forces"])
    
    num_python_subprocesses: int = zntrack.meta.Text(1)

    # def _post_init_(self):
    # self.calculator = utils.get_deps_if_node(self.calculator, "calc")
    # TODO: base class with ASE calculators

    def run(self):
        if isinstance(self.calculator_node, GAP):
            calculator = (
                quippy.potential.Potential,
                [],
                {"param_filename": self.calculator_node.model_xml_file.as_posix()},
            )
        else:
            raise NotImplementedError

        inputs = ConfigSet([x for x in self.data])
        outputs = OutputSpec()
        
        config_set = generic.run(
            inputs=inputs,
            calculator=calculator,
            outputs=outputs,
            output_prefix=None,
            properties=self.properties,
            autopara_info={"num_python_subprocesses": self.num_python_subprocesses}
            # "autopara_info.remote_info" for remote execution # configuration file for cluster
        )
        # TODO consider using ase info and arrays field instead of calculators
        # requires ``export OMP_NUM_THREADS=1``

        self.atoms = list(config_set)
