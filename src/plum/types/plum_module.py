from .plum_object import PlumObject
from .property import Parameter as P, Hyperparameter as HP, Submodule as SM
import torch.nn as nn
try:
    import ujson as json
except ModuleNotFoundError:
    import json


class PlumModule(PlumObject, nn.Module): 
    def __init__(self, *args, **kwargs):

        self._parameter_names = {}
        self._parameter_tags = {}
        self._submodule_names = {}
        self._submodule_tags = {}

        nn.Module.__init__(self)
        PlumObject.__init__(self, *args, **kwargs)

    def _initialize_plum_properties(self, kwargs):
        super(PlumModule, self)._initialize_plum_properties(kwargs)

        for name, param in P.iter_named_parameters(self):
            self._add_parameter(name, param)

        for name, submodule in SM.iter_named_submodules(self):
            self._add_submodule(name, submodule, kwargs.get(name, None))
            if name in kwargs:
                del kwargs[name]

    def _add_parameter(self, name, param_property):
        param = param_property.new(self)
        self._parameters[name] = param
        self._parameter_names[param_property] = name
        self._parameter_tags[name] = param_property.tags

    def _add_submodule(self, name, submod_property, value):
        self._modules[name] = submod_property.new(self, value)
        self._submodule_names[submod_property] = name
        self._submodule_tags[name] = submod_property.tags
    
    def parameter_tags(self, name):
        if "." in name:
            getters = name.split(".")
            owner = self
            for getter in getters[:-1]:
                owner = getattr(owner, getter)
            if hasattr(owner, "parameter_tags"):
                return owner.parameter_tags(getters[-1])
            elif hasattr(owner, "_parameter_tags"):
                return owner._parameter_tags[getters[-1]]
            else:
                return ()
        else:
            return self._parameter_tags.get(name, None)

    def to_json(self, to_string=True):
        r = {}
        r["__plum_type__"] = self.plum_id
        
        for name, _ in HP.iter_named_hyperparameters(self):
            value = getattr(self, name)
            r[name] = _to_json_helper(value)

        for name, _ in SM.iter_named_submodules(self):
            value = getattr(self, name)
            if isinstance(value, nn.ModuleList):
                r[name] = [_to_json_helper(x) for x in value]
            elif isinstance(value, nn.ModuleDict):
                r[name] = {x: _to_json_helper(y) for x, y in value.items()}
            else:
                r[name] = _to_json_helper(value)
        if to_string:
            r = json.dumps(r)
        return r

def _to_json_helper(obj):
    if isinstance(obj, (list, tuple)):
        return [_to_json_helper(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _to_json_helper(v) for k, v in obj.items()}
    elif hasattr(obj, "to_json"):
        return obj.to_json(to_string=False)
    else:
        return obj
