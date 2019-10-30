



class PlumProperty:
    def __init__(self, doc=None):
        if doc is not None:
            self.__doc__ = doc

    def __get__(self, owner):
        raise NotImplemented()

    @staticmethod
    def baseclasses(cls):

        results = set(cls.__bases__)
        for b in cls.__bases__:
            results.update(PlumProperty.baseclasses(b))
        return results

    @classmethod
    def iter_named_plum_property(self, plum_module, prop_type=None):
        
        if prop_type is None:
            prop_type = PlumProperty

        classes = PlumProperty.baseclasses(plum_module.__class__)
        classes.add(type(plum_module))

        for cls in classes:
            for name, obj in cls.__dict__.items():
                if isinstance(obj, prop_type):
                    yield name, obj
    @classmethod
    def iter_plum_property(self, plum_module, prop_type=None):
        
        if prop_type is None:
            prop_type = PlumProperty

        classes = PlumProperty.subclasses(type(plum_module))
        classes.add(type(plum_module))

        for cls in classes:
            for obj in cls.__dict__.values():
                if isinstance(obj, prop_type):
                    yield obj
