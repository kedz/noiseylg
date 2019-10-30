from plum.types import register, PlumModule, HP, props

from subprocess import check_output
from queue import Queue
from threading import Thread

from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import d2t.preprocessing.tvs as preproc



@register("metrics.tv_metrics")
class TVMetrics(PlumModule):

    path = HP(type=props.EXISTING_PATH)
    search_fields = HP()
    references_fields = HP()

    def __pluminit__(self):
        self._cache = None
        self._queue = Queue(maxsize=0)
        self._thread = None
        self._thread = Thread(target=self._process_result)
        self._thread.setDaemon(True)
        self._thread.start()
        self._hyp_fp = NamedTemporaryFile("w")
        self._ref_fp = NamedTemporaryFile("w")

    def postprocess(self, tokens, mr):
        # TODO right now this is specific to the e2e dataset. Need to 
        # generalize how to do post processing. 
        tokens = [t for t in tokens if t[0] != "<" and t[-1] != ">"]
        text = " ".join(tokens)
        return preproc.lexicalize(text, mr)



    def _process_result(self):
        while True:
            hyp, refs, mr = self._queue.get()

            print(self.postprocess(hyp, mr), file=self._hyp_fp)
            #print(" ".join(hyp), file=self._hyp_fp)
            
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

    def apply_fields(self, fields, obj):
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if hasattr(field, "__call__"):
                obj = field(obj)
            else:
                obj = obj[field]
        return obj

    def forward(self, forward_state, batch):
        search = self.apply_fields(self.search_fields, forward_state)
        hypotheses = search.output()
        reference_sets = self.apply_fields(self.references_fields, batch)

        for i, (hyp, refs) in enumerate(zip(hypotheses, reference_sets)):
            self._queue.put([hyp, refs, batch["mr"][i]])

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
