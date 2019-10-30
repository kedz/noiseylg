from .plum_property import PlumProperty
import torch
import torch.nn as nn


class Parameter(PlumProperty):

    def __init__(self, *dim_names, type=torch.FloatTensor, tags=None,
                 conditional=None, doc=None):
        super(Parameter, self).__init__(doc=doc)

        if len(dim_names) == 1 and hasattr(dim_names[0], "__call__"):
            self._dim_names = ()
            self._dim_func = dim_names[0]
        else:
            self._dim_names = tuple(dim_names)
            self._dim_func = None

        if tags is None:
            self._tags = ()
        elif isinstance(tags, (list, tuple)):
            self._tags = tuple(tags)
        elif isinstance(tags, str):
            self._tags = tuple([tags])
        else:
            raise ValueError(
                "tags must be None, a str, or a list/tuple of str")
        self._type = type
        self._conditional = conditional

    @property
    def tags(self):
        return self._tags

    @property
    def dim_names(self):
        return self._dim_names

    @property
    def dim_func(self):
        return self._dim_func

    @property
    def type(self):
        return self._type

    @property
    def conditional(self):
        return self._conditional

    def new(self, owner):
        if self.conditional is None or getattr(owner, self.conditional):

            if self.dim_func is not None:
                dim_sizes = self.dim_func(owner)
            else:
                dim_sizes = [getattr(owner, dim_name) 
                             for dim_name in self.dim_names]
            parameter = nn.Parameter(self.type(*dim_sizes).normal_())
            return parameter
        else:
            return None

    def __get__(self, obj, objtype=None):
        return obj._parameters[obj._param_names[self]]

    @classmethod
    def iter_named_parameters(cls, plum_module):
        return cls.iter_named_plum_property(plum_module, prop_type=cls)

    @classmethod
    def iter_parameters(cls, plum_module):
        return cls.iter_plum_property(plum_module, prop_type=cls)
