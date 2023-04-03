import ipsuite as ips


def test_ase_md(proj_path, traj_file):
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        model = ips.models.GAP(data=data.atoms)
        md = ips.calculators.ASEMD(
            data=data.atoms,
            model=model,
            temperature=1,
            time_step=1,
            friction=1,
            steps=100,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run()

    md.load()

    assert len(md.atoms) == 100
