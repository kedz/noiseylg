class PlumObjectRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, plum_id, clazz):
        if plum_id in self._registry:
            raise ValueError(
                'plum_id "{}" is already assigned to class {}'.format(
                    plum_id, self._registry[plum_id].__name__))
        else:
            self._registry[plum_id] = clazz

    def new_object(self, plum_id, kwargs):
        if plum_id not in self._registry:
            raise ValueError('plum_id "{}" is not registered.'.format(plum_id))
        return self._registry[plum_id](**kwargs)

PLUM_OBJECT_REGISTRY = PlumObjectRegistry()
