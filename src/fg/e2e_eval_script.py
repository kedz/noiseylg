from plum.types import register, PlumModule, HP, props
from plum.utils import resolve_getters

from subprocess import check_output
from queue import Queue
from threading import Thread

from pathlib import Path
from tempfile import NamedTemporaryFile
import json

from d2t.postedit import e2e as postedit


@register("d2t.metrics.e2e_eval_script")
class E2EEvalScript(PlumModule):

    path = HP(type=props.EXISTING_PATH)
    search_fields = HP()
    references_fields = HP()
    labels_fields = HP()

    def __pluminit__(self):
        self._cache = None
        self._queue = Queue(maxsize=0)
        self._thread = None
        self._thread = Thread(target=self._process_result)
        self._thread.setDaemon(True)
        self._thread.start()
        self._hyp_fp = NamedTemporaryFile("w")
        self._ref_fp = NamedTemporaryFile("w")

    def _process_result(self):
        while True:
            hyp, refs, labels = self._queue.get()
            hyp = postedit.detokenize(hyp)
            hyp = postedit.lexicalize(hyp, labels)
            print(hyp, file=self._hyp_fp)
            
            if isinstance(refs, (list, tuple)):
                refs = "\n".join(refs)
            
            print(refs, file=self._ref_fp, end="\n\n")

            self._queue.task_done()

    def reset(self):
        self._cache = None
        while not self._queue.empty():
            self._queue.get()
            self._queue.task_done()
        self._hyp_fp = NamedTemporaryFile("w")
        self._ref_fp = NamedTemporaryFile("w")

    def forward(self, forward_state, batch):
        search = resolve_getters(self.search_fields, forward_state)
        hypotheses = search.output()
        reference_sets = resolve_getters(self.references_fields, batch)
        all_labels = resolve_getters(self.labels_fields, batch)

        for hyp, refs, labels in zip(hypotheses, reference_sets, all_labels):
            self._queue.put([hyp, refs, labels])

    def run_script(self):

        self._queue.join()

        self._ref_fp.flush()
        self._hyp_fp.flush()

        script_path = Path(self.path).resolve()
        result_bytes = check_output(
            [str(script_path), self._hyp_fp.name, self._ref_fp.name])
        result = json.loads(result_bytes.decode("utf8"))
        self._cache = result

        self._ref_fp = None
        self._hyp_fp = None

    def compute(self):
        if self._cache is None:
            self.run_script()
        return self._cache

    def pretty_result(self):
        return str(self.compute())
