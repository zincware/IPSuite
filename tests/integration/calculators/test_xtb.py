import ipsuite as ips


def test_xtb(data_repo):
    data = ips.AddData.from_rev(name="BMIM_BF4_363_15K")
    with ips.Project() as proj:
        confs = ips.configuration_selection.IndexSelection(data=data, indices=[100])
        conf = ips.calculators.xTBSinglePoint(data=confs.atoms)
    proj.run()
