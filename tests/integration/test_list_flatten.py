import ase.io
import numpy as np
import numpy.testing as npt
import zntrack as zn
from ase import units

import ipsuite as ips
from ipsuite.utils.ase_sim import get_density_from_atoms


def test_ase_md(proj_path, cu_box):
    atoms = []
    for _ in range(5):
        atoms.extend(cu_box)

    ase.io.write("cu_box.xyz", atoms)
    check = ips.TemperatureCheck()
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    rescale_box = ips.RescaleBoxModifier(cell=10)
    temperature_ramp = ips.TemperatureRampModifier(temperature=300)
    model = ips.EMTSinglePoint()
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        mapped_md = zn.apply(ips.ASEMD, method="map")(
            data=data.atoms,
            data_ids=[0, 1, 2],
            model=model,
            checks=[check],
            modifiers=[rescale_box, temperature_ramp],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )
        flat_md = ips.Flatten(data=mapped_md.structures)

    project.repro()

    assert len(mapped_md.atoms) == len(flat_md.atoms)
