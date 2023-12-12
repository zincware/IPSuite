import ipsuite as ips
import requests


def test_md22():
    for url in ips.datasets.MD22Dataset.datasets.values():
        response = requests.get(url)
        assert response.status_code == 200

    project = ips.Project(automatic_node_names=True)

    with project:
        data = ips.datasets.MD22Dataset("AT-AT")

    project.run()

    data.load()
    assert len(data.atoms) > 0

