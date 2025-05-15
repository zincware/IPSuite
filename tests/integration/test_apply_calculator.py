
import ipsuite as ips

def test_apply_calculator(proj_path):
    project = ips.Project()
    
    model = ips.MACEMPModel()

    with project:
        water = ips.Smiles2Conformers(
            smiles="O",
            numConfs=10,
        )
        traj = ips.ApplyCalculator(
            data=water.frames,
            model=model,
        )
    
    project.repro()

    assert len(traj.frames) == 10

