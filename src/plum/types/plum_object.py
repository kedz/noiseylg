import inspect
from .property import Hyperparameter as HP, Submodule as SM
from pathlib import Path
try:
    import ujson as json
except ModuleNotFoundError:
    import json
import torch


class PlumObject(object):

    def __init__(self, *args, **kwargs):
        self._hyperparameters = {}
        self._hyperparameter_tags = {}
        if len(args) != 0:
            raise ValueError("PlumObject constructors use kwargs only.")

        self._initialize_plum_properties(kwargs)
        # _initialize_plum_properties consumes key/values in kwargs.
        # After calling, kwargs should be empty.
        # Raise exception if kwargs contains extraneous arguments just to be
        # safe.
        for arg in kwargs:
            raise ValueError("{} has no constructor argument {}".format(
                self.__class__.__name__, arg))

        pluminit_args = self._build_pluminit_args()            
        self.__pluminit__(*pluminit_args)

    def _initialize_plum_properties(self, kwargs):
        for name, hyperparameter in HP.iter_named_hyperparameters(self):
            prop_value = kwargs.get(name, None)
            self._add_hyperparameter(name, hyperparameter, prop_value)
            if name in kwargs:
                del kwargs[name]

    def _build_pluminit_args(self):
        # Build arguments for subclass __pluminit__ method and check for
        # arguments that are not hyperparameters.
        args = []
        for argname in inspect.getfullargspec(self.__pluminit__)[0][1:]:
            if not hasattr(self, argname):
                raise Exception(
                    '__pluminit__ argument "{}" is not a ' \
                    'property of class {}.{}'.format(
                        argname, self.__class__.__module__,
                        self.__class__.__name__))
            args.append(getattr(self, argname))
        return args

    def _add_hyperparameter(self, name, hparam_property, value):
        self._hyperparameters[hparam_property] = hparam_property.new(
            self, name, value)
        self._hyperparameter_tags[name] = hparam_property.tags

    def __pluminit__(self):
        pass

    @property
    def plum_id(self):
        return self.__class__.__dict__.get("__plumid__", None)

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

    def save(self, path):
        if isinstance(path, str):
            path = Path(path)
            
        if not path.parent.exists():
            path.parent.mkdir(exist_ok=True, parents=True)

        plum_data = self.to_json()
        if hasattr(self, "state_dict"):
            state_dict = self.state_dict()
        else:
            state_dict = None

        torch.save({"plum_data": plum_data, "state_dict": state_dict}, path)

def _to_json_helper(obj):
    if isinstance(obj, (list, tuple)):
        return [_to_json_helper(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _to_json_helper(v) for k, v in obj.items()}
    elif hasattr(obj, "to_json"):
        return obj.to_json(to_string=False)
    else:
        return obj
