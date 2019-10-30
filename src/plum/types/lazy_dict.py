

class LazyDict(dict):

    def __init__(self, *args, **kwargs):
        self._closures = set()
        super(LazyDict, self).__init__(*args, **kwargs)

    def lazy_set(self, key, func):
        
        self[key] = func
        self._closures.add(key)

    def __getitem__(self, key):
        if key in self._closures:
            val = super(LazyDict, self).__getitem__(key)()
            self._closures.remove(key)
            super(LazyDict, self).__setitem__(key, val)
            return val
        else:
            return super(LazyDict, self).__getitem__(key)

    def __setitem__(self, key, value):

        if key in self._closures:
            self._closures.remove(key)

        super(LazyDict, self).__setitem__(key, value)

    def update(self, other):
        raise Exception("Implement me")
