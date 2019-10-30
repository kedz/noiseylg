from ..types import register, PlumObject, HP, props


@register("dataio.stack_ds")
class StackDatasource(PlumObject):
   
    datasources = HP()


    def __getitem__(self, index):

        if index < 0:
            if -index > len(self):
                raise IndexError("datasource index out of range")
            index = len(self) + index

        i = 0
        n = 0 
        offset = 0
        while i < len(self.datasources):

            n += len(self.datasources[i])

            if index < n:
                return self.datasources[i][index - offset]
            
            i += 1
            offset = n
        raise IndexError("datasource index out of range")

    def __len__(self):
        return sum([len(ds) for ds in self.datasources])

    @property
    def paths(self):
        paths = []
        for ds in self.datasources:
            for path in ds.paths:
                paths.append(path)
        return paths

    @property
    def path(self):
        return self.paths
