from .plum_object import PlumObject
from .object_registry import PlumObjectRegistry, PLUM_OBJECT_REGISTRY
from .plum_module import PlumModule
from .property import Hyperparameter as HP, Parameter as P, Submodule as SM
from . import property_types as props
from .lazy_dict import LazyDict
from .variable import Variable


def register(plum_id, registry=None):
    if registry is None:
        registry = PLUM_OBJECT_REGISTRY

    def wrapper(cls):
        cls.__plumid__ = plum_id
        registry.register(plum_id, cls)
        return cls
    return wrapper
