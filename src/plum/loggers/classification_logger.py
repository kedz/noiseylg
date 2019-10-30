from pathlib import Path
from plum.types import register, PlumObject, HP, props
from plum.utils import resolve_getters


@register("plum.loggers.classification_logger")
class ClassificationLogger(PlumObject):

    file_prefix = HP(type=props.STRING)
    input_fields = HP()
    output_fields = HP()
    target_fields = HP(required=False)
    vocab = HP(required=False)
    log_every = HP(default=1, type=props.INTEGER)

    def __pluminit__(self):
        self._epoch = 0
        self._log_dir = None
        self._file = None
        self._fp = None
        self._steps = 0

    def set_log_dir(self, log_dir):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True, parents=True)

    def __call__(self, forward_state, batch):

        self._steps += 1
        if self._steps % self.log_every != 0:
            return
        ref_inputs = resolve_getters(self.input_fields, batch)
        pred_labels = resolve_getters(self.output_fields, forward_state)\
            .tolist()

        target = resolve_getters(self.target_fields, batch)

        for i, (ref_input, pred_label) in enumerate(
                    zip(ref_inputs, pred_labels)):

            if not isinstance(pred_label, str) and self.vocab is not None:
                pred_label = self.vocab[pred_label]

            print("input: {}".format(ref_input), file=self._fp) 
            if target is not None:

                true_label = target[i]
                
                if not isinstance(true_label, str) and self.vocab is not None:
                    true_label = self.vocab[true_label]

                print("pred_label: {} target_label: {}".format(
                    pred_label, true_label), file=self._fp)
            else:
                print("pred_label: {}".format(pred_label), file=self._fp)
            print(file=self._fp)
            
    def next_epoch(self):
        
        self.close()

        self._steps = 0
        self._epoch += 1
        self._file = self._log_dir / "{}.{}.log".format(
            self.file_prefix, self._epoch)
        self._fp = self._file.open("w")
         
    def close(self):
        if self._fp:
            self._fp.close()

    def __del__(self):
        self.close()
