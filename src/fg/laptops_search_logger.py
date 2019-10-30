from plum.types import register, PlumObject, HP, props
from plum.utils import resolve_getters
import torch
from pathlib import Path
import d2t.preprocessing.laptops as preproc


@register("fg.loggers.laptops_search_logger")
class LaptopsSearchLogger(PlumObject):

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

    def __call__(self, forward_state, batch):
        
        search = resolve_getters(self.search_fields, forward_state)
        inputs = resolve_getters(self.input_fields, batch)
        references = resolve_getters(self.reference_fields, batch)
 
        for i, output in enumerate(search.output()):
            if inputs:
                print("inputs:", file=self._fp)
                #print(inputs[i], file=self._fp)
                print(batch["mr"][i], file=self._fp)
            if references:
                print("references:", file=self._fp)
                if isinstance(references[i], (list, tuple)):
                    print("\n".join(references[i]), file=self._fp)
                else:
                    print(references[i], file=self._fp)
            print("hypothesis:", file=self._fp)
            print(preproc.lexicalize(" ".join(output), inputs[i]), 
                  file=self._fp)
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
