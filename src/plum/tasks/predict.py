from ..types import register, PlumObject, HP, props
import plum


def postprocess(tokens):
    import re
    
    tokens = [t for t in tokens if t[0] != "<" and t[-1] != ">"]
    text = " ".join(tokens)
    text = re.sub(r" (\.|,)", r"\1", text) 
    text = re.sub(r" - ", r"-", text) 
    text = re.sub("\u00A3 ", "\u00A3", text) 
    return text

@register("tasks.predict")
class Predict(PlumObject):

    checkpoint = HP(required=False)

    batches = HP()
    
    reference_fields = HP(required=False)
    output_fields = HP()
    
    def apply_fields(self, fields, obj):
        if not isinstance(fields, (list, tuple)):
            fields = [fields]

        for field in fields:
            if hasattr(field, "__call__"):
                obj = field(obj)
            else:
                obj = obj[field]
        return obj

    def run(self, env, verbose=False):
        ckpt, ckpt_md = self.get_checkpoint(env)

        if verbose:
            print("Reading checkpoint from {}".format(ckpt_md["path"]))
        model = plum.load(ckpt_md["path"]).eval()

        from plum.seq2seq.search import BeamSearch

        model.search_algos["beam"] = BeamSearch(
            max_steps=200, beam_size=4,
            vocab=model.search_algos["greedy"].vocab)

        if env["gpu"] > -1:
            if verbose:
                print("Placing model and batches on device {}".format(
                    env["gpu"]))
            model.cuda(env["gpu"])
            self.batches.gpu = env["gpu"]

        for batch in self.batches:
            state = model(batch)
            if self.reference_fields:
                refs = self.apply_fields(self.reference_fields, batch)
            else:
                refs = None
            pred = self.apply_fields(self.output_fields, state)
            outputs = pred.output()
            for i, output in enumerate(outputs):
                if refs:
                    if isinstance(refs[i], (list, tuple)):
                        print("\n".join(refs[i]))
                    else:
                        print(refs[i])
                print(postprocess(output))
                print()

    def get_checkpoint(self, env):

        if len(env["checkpoints"]) == 0:
            raise RuntimeError("No model checkpoints!")

        if self.checkpoint is not None:
            if self.checkpoint not in env["checkpoints"]:
                raise RuntimeError("Checkpoint {} does not exist.".format(
                    self.checkpoint))
            else:
                return self.checkpoint, env["checkpoints"][self.checkpoint]

        else:
            for ckpt, md in env["checkpoints"].items():
                if md.get("default", False):
                    return ckpt, md
            return ckpt, md
