from .plum_property import PlumProperty


def any_type(x):
    return True


class Hyperparameter(PlumProperty):

    def __init__(self, default=None, required=True, tags=None, 
                 type=None, doc=None):
        super(Hyperparameter, self).__init__(doc=doc)
        self._default = default
        self._required = required
        self._tags = tags
        if type is None:
            self._type = any_type
        else:
            self._type = type

    @property
    def default(self):
        return self._default

    @property
    def required(self):
        return self._required

    @property
    def tags(self):
        return self._tags

    @property
    def type(self):
        return self._type

    def new(self, owner, name, value):

        if value is None:
            value = self.default

        if self.required and value is None:
            raise ValueError("{}.{} missing required argument {}".format(
                owner.__class__.__module__, owner.__class__.__name__, name))
        elif not self.type(value):
            raise ValueError("Bad type for {}.{} argument {}: {}".format(
                owner.__class__.__module__, owner.__class__.__name__, name,
                str(value)))
        else:
            return value
 
    def __get__(self, owner, owner_type=None):
        return owner._hyperparameters[self]

    @classmethod
    def iter_named_hyperparameters(cls, plum_module):
        return cls.iter_named_plum_property(plum_module, prop_type=cls)

    @classmethod
    def iter_hyperparameters(cls, plum_module):
        return cls.iter_plum_property(plum_module, prop_type=cls)
