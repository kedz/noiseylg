from ..types import register, PlumObject, HP, props

import pandas as pd


@register("dataio.csv")
class CSV(PlumObject):

    path = HP(type=props.EXISTING_PATH)
    sep = HP(type=props.STRING)
    header = HP(type=props.BOOLEAN)

    @property
    def header_names(self):
        return self._header_names

    def __pluminit__(self, path):
        header = 'infer' if self.header else None
        self._dataframe = pd.read_csv(path, sep=self.sep, header=header)

    def __getitem__(self, index):
        return self._dataframe.iloc[index].to_dict()

    def __len__(self):
        return len(self._dataframe)

    def __repr__(self):
        return "dataio.CSV({}, header={}, sep='{}', {} rows)".format(
            self.path, self.header, self.sep, len(self))

    @property
    def paths(self):
        return [self.path]
