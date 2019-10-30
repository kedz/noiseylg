from .plum_property import PlumProperty
import torch.nn as nn


class Submodule(PlumProperty):

    def __init__(self, default=None, required=True, type=None, tags=None):
        self._default = default
        self._required = required
    
        if type is None:
            def any_type(x):
                return True
            self._type = any_type
        elif hasattr(type, "__call__"):
            self._type = type
        else:
            raise ValueError("type must be None or implement __call__")

        if isinstance(tags, str):
            self._tags = tuple([tags])
        elif isinstance(tags, (list, tuple)):
            self._tags = tuple(tags)
        elif tags is None:
            self._tags = tuple()
        else:
            raise ValueError(
                "tags must be None, a str, or a list/tuple of str")

    @property
    def default(self):
        return self._default

    @property
    def required(self):
        return self._required

    @property
    def type(self):
        return self._type

    @property
    def tags(self):
        return self._tags

    def new(self, owner_module, submodule):

        if submodule is None:
            submodule = self.default

        if submodule is None:
            if not self.required:
                return None  
            else:
                raise Exception("Missing submodule for {}".format(
                    str(owner_module.__class__)))

        elif isinstance(submodule, (list, tuple)):
            for subsubmod in submodule:
                if not self.type(subsubmod) or \
                        not issubclass(subsubmod.__class__, nn.Module):
                    raise ValueError("Bad type: {}".format(type(subsubmod)))
            return nn.ModuleList(submodule)
        elif isinstance(submodule, dict):
            for subsubmod in submodule.values():
                if not self.type(subsubmod) or \
                        not issubclass(subsubmod.__class__, nn.Module):
                    raise ValueError("Bad type: {}".format(type(subsubmod)))
                
            return nn.ModuleDict(submodule)
        else:
            if not issubclass(submodule.__class__, nn.Module):
                raise ValueError("Bad type: {}".format(type(submodule)))
            return submodule

    def __get__(self, owner_module, owner_type=None):
        return owner_module._modules[owner_module._submodule_names[self]]

    @classmethod
    def iter_named_submodules(cls, plum_module):
        return cls.iter_named_plum_property(plum_module, prop_type=cls)

    @classmethod
    def iter_submodules(cls, plum_module):
        return cls.iter_plum_property(plum_module, prop_type=cls)
