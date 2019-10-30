from ..types import register, PlumObject, HP, props


@register("dataio.parallel_datasources")
class ParallelDatasources(PlumObject):
   
    datasources = HP()

    def __pluminit__(self, datasources):
        len0 = len(datasources[0])
        for ds in datasources[1:]:
            if len(ds) != len0:
                raise RuntimeError(
                    "Datasources must have the same number of datapoints. " + \
                    "Expected {} datapoints but found {} in {}".format(
                        len0, len(ds), str(ds)))

    def __getitem__(self, index):
        return [ds[index] for ds in self.datasources]

    def __len__(self):
        return len(self.datasources[0])

    def __repr__(self):
        return "datio.parallel({}, {} lines)".format(
            ", ".join([ds.plum_id for ds in self.datasources]), len(self))

    @property
    def paths(self):
        paths = []
        for ds in self.datasources:
            for path in ds.paths:
                paths.append(path)
        return paths

