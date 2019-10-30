from ..types import register, PlumObject, HP, props
try:
    import ujson as json
except ModuleNotFoundError:
    import json
import mmap
import contextlib


@register("dataio.mmap_jsonl")
class MMAPJSONL(PlumObject):
    
    path = HP(type=props.EXISTING_PATH)
    
    def __pluminit__(self, path):
        with open(path, "r") as fp:
            self._mmap = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        self._offsets = self._build_byte_offsets()

    def _build_byte_offsets(self):
        offsets = [] 
        start = 0
        stop = self._mmap.find(b'\n')
        while stop != -1:
            offsets.append([start, stop])
            start = stop + 1
            stop = self._mmap.find(b'\n', start)

        file_size = self._mmap.size()
        delta = int(file_size) - int(offsets[-1][-1] + 1)
        if delta > 0:
            offsets.append([offsets[-1][-1] + 1, file_size])

        return offsets

    def __getitem__(self, index):
        start, stop = self._offsets[index]
        size = stop - start
        self._mmap.seek(start)
        raw_bytes = self._mmap.read(size)
        data = json.loads(raw_bytes.decode("utf8"))
        return data

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        return "datio.JSONL({}, {} lines, mmap)".format(self.path, len(self))

    @property
    def paths(self):
        return [self.path]

    def __del__(self):
        self._mmap.close()
