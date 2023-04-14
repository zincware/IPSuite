class _Nodes:
    GAP = "ipsuite.models.GAP"


def __getattr__(name):
    import importlib

    _name = getattr(_Nodes, name)

    module, class_name = _name.rsplit(".", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)


def __dir__():
    return [name for name in dir(_Nodes) if not name.startswith("_")]


__all__ = __dir__()
