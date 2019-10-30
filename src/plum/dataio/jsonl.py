from ..types import register, PlumObject, HP, props
try:
    import ujson as json
except ModuleNotFoundError:
    import json


@register("dataio.jsonl")
class JSONL(PlumObject):
    
    path = HP(type=props.EXISTING_PATH)

    def __pluminit__(self, path):
        self._data = []
        with open(path, "r") as fp:
            for line in fp:
                self._data.append(json.loads(line))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "datio.JSONL({}, {} lines)".format(self.path, len(self))

    @property
    def paths(self):
        return [self.path]

