from ipsuite import __all__


def nodes() -> dict[str, list[str]]:
    """Return all available nodes."""
    return {"ipsuite": __all__}
