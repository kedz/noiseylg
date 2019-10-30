from plum.types import register, PlumModule, HP, props, Variable
from plum.utils import resolve_getters
import torch


@register("fg.metrics.sequence_classification_error")
class SequenceClassificationError(PlumModule):

    input_vocab = HP()
    classifier = HP()
    gpu = HP(default=-1)
    target_fields = HP()
    search_fields = HP()

    def __pluminit__(self, classifier, gpu):
        if gpu > -1:
            classifier.cuda(gpu)

    def reset(self):
        self._errors = 0
        self._total = 0

    def _make_classifier_inputs(self, batch):
        lens = []
        clf_inputs = []
        for out in batch:
            clf_inputs.append(
                [self.input_vocab.start_index] 
                    + [self.input_vocab[t] for t in out[:-1]]
                    + [self.input_vocab.stop_index]
            )
            lens.append(len(out) + 1)
        lens = torch.LongTensor(lens)
        max_len = lens.max().item()
        clf_inputs = torch.LongTensor(
            [inp + [self.input_vocab.pad_index] * (max_len - len(inp))
              for inp in clf_inputs]
        ).t()

        clf_inputs = Variable(
            clf_inputs,
            lengths=lens,
            batch_dim=1, length_dim=0,
            pad_value=self.input_vocab.pad_index)
        if self.gpu > -1:
            return clf_inputs.cuda(self.gpu)
        else:
            return clf_inputs
    
    def forward(self, forward_state, batch):

        self.classifier.eval()
        search_outputs = resolve_getters(
            self.search_fields, forward_state).output()
        clf_inputs = self._make_classifier_inputs(search_outputs)

        fs = self.classifier({"inputs": clf_inputs})
        targets = resolve_getters(self.target_fields, batch)

        if self.gpu > -1:
            targets = targets.cuda(self.gpu)

        errors = (fs["output"] != targets).long().sum().item()
        total = targets.size(0)

        self._errors += errors
        self._total += total

    def compute(self):
        return {"rate": self._errors / self._total if self._total > 0 else 0.,
                "count": self._errors}

    def pretty_result(self):
        return str(self.compute())
