import requests

import ipsuite as ips


def test_md22(proj_path):
    for url in ips.MD22Dataset.datasets.values():
        response = requests.get(url)
        assert response.status_code == 200

    project = ips.Project()

    with project:
        data = ips.MD22Dataset(dataset="AT-AT")

    project.repro()

    assert len(data.atoms) > 0
