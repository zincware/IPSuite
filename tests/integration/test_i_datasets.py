import requests

import ipsuite as ips


def test_md22():
    for url in ips.datasets.MD22Dataset.datasets.values():
        response = requests.get(url)
        assert response.status_code == 200

    project = ips.Project()

    with project:
        data = ips.datasets.MD22Dataset(dataset="AT-AT")

    project.run()

    assert len(data.atoms) > 0
