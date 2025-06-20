import ipsuite as ips


def test_Smiles2Atoms(proj_path):
    project = ips.Project()

    d3 = ips.TorchDFTD3(
        xc="pbe",
        damping="bj",
        cutoff=3,
        cnthr=3,
        abc=False,
        dtype="float32",
    )

    with project:
        data = ips.Smiles2Atoms(smiles="CCO")
        data_with_d3 = ips.ApplyCalculator(
            data=data.frames, model=d3
        )

    project.repro()

    assert data_with_d3.frames[0].get_potential_energy() < 0
    assert data_with_d3.frames[0].get_forces().shape == (9, 3)
