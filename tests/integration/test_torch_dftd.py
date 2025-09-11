import ipsuite as ips


def test_torch_dftd3(proj_path):
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


def test_torch_dft3_and_mp0(proj_path):
    project = ips.Project()

    d3 = ips.TorchDFTD3(
        xc="pbe",
        damping="bj",
        cutoff=3,
        cnthr=3,
        abc=False,
        dtype="float32",
    )

    mp0 = ips.MACEMPModel()

    with project:
        data = ips.Smiles2Atoms(smiles="CCO")
        mix_calc = ips.MixCalculator(
            calculators=[d3, mp0],
        )

        data_with_d3 = ips.ApplyCalculator(
            data=data.frames, model=mix_calc
        )

    project.repro()

    assert data_with_d3.frames[0].get_potential_energy() < 0
    assert data_with_d3.frames[0].get_forces().shape == (9, 3)
