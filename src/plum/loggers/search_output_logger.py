from pathlib import Path
from plum.types import register, PlumObject, HP, props


@register("loggers.search_output_logger")
class SearchOutputLogger(PlumObject):

    file_prefix = HP(type=props.STRING)
    search_fields = HP()
    input_fields = HP(required=False)
    reference_fields = HP(required=False)

    def __pluminit__(self):
        self._epoch = 0
        self._log_dir = None
        self._file = None
        self._fp = None

    def set_log_dir(self, log_dir):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True, parents=True)

    def _apply_fields(self, item, fields):
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if hasattr(field, "__call__"):
                item = field(item)
            else:
                item = item[field]
        return item

    def __call__(self, forward_state, batch):
        
        search = self._apply_fields(forward_state, self.search_fields)
        if self.input_fields:
            inputs = self._apply_fields(batch, self.input_fields)
        else:
            inputs = None

        if self.reference_fields:
            references = self._apply_fields(batch, self.reference_fields)
        else:
            references = None
 
        for i, output in enumerate(search.output()):
            if inputs:
                print("inputs:", file=self._fp)
                print(inputs[i], file=self._fp)
            if references:
                print("references:", file=self._fp)
                print(references[i], file=self._fp)
            print("hypothesis:", file=self._fp)
            print(" ".join(output), file=self._fp)
            print(file=self._fp)
            
    def next_epoch(self):
        
        self.close()

        self._epoch += 1
        self._file = self._log_dir / "{}.{}.log".format(
            self.file_prefix, self._epoch)
        self._fp = self._file.open("w")
         
    def close(self):
        if self._fp:
            self._fp.close()
