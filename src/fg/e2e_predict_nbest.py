import plum
from plum.types import register, PlumObject, HP, SM, props, Variable
import torch
import numpy as np
from itertools import combinations, product
from scipy.special import comb
import json
from d2t.postedit import e2e as postedit

@register("d2t.laptop_predict_nbest")
class LaptopPredictNBEST(PlumObject):
 
    checkpoint = HP(required=False)
    beam_size = HP(default=1, type=props.INTEGER)
    source_vocab = HP()
    target_vocab = HP()
    input_path = HP(type=props.EXISTING_PATH)
    filename = HP()
    nbest = HP()

    def run(self, env, verbose=False):

        output_path = env["proj_dir"] / "output" / self.filename
        output_path.parent.mkdir(exist_ok=True, parents=True)

        if self.checkpoint is None:
            ckpt = self._get_default_checkpoint(env)
        else:
            ckpt = self.checkpoint
        if ckpt is None:
            raise RuntimeError("No checkpoints found!")
            
        ckpt_path = env["checkpoints"][ckpt]["path"]
        if verbose:
            print("Loading model from {}".format(ckpt_path))
        model = plum.load(ckpt_path).eval()
        if env["gpu"] > -1:
            model.cuda(env["gpu"])
        self._gpu = env["gpu"]

        with open(self.input_path, 'r') as fp, \
                open(output_path, "w") as out_fp:
            for line in fp:

                data = json.loads(line)
                gen_input = self.make_generator_inputs(data)
                outputs = self._get_outputs(model, gen_input)
                nbest = []
                for out in outputs:
                    nbest.append({"tokens": out[:-1], 
                                  "text": " ".join(out[:-1])})
                data = json.dumps({
                    "mr": data["mr"],
                    "nbest": nbest,
                    #"tokens": tokens,
                    #"text": text,
                })
                print(data, file=out_fp, flush=True)

    def make_generator_inputs(self, data):
        tokens = [self.source_vocab.start_token] + data["source"] \
            + [self.source_vocab.stop_token]
        inputs = Variable(
            torch.LongTensor([[self.source_vocab[t] for t in tokens]]).t(),
            lengths=torch.LongTensor([len(tokens)]),
            length_dim=0, batch_dim=1, 
            pad_value=self.source_vocab.pad_index) 
        if self._gpu > -1:
            inputs = inputs.cuda(self._gpu)
        return {"source_inputs": inputs}


    def _get_outputs(self, model, inputs):
        state = model.encode(inputs)
        search = plum.seq2seq.search.BeamSearch(
            max_steps=100, beam_size=self.beam_size,
            vocab=self.target_vocab)
        search(model.decoder, state)
        outputs = search.output(n_best=self.nbest)
        return outputs[0]
        #raw_tokens = outputs[0][:-1]
        #text = " ".join(raw_tokens)
        #postedited_output = postedit.detokenize(raw_tokens)
        #postedited_output = postedit.lexicalize(postedited_output, labels)

        return raw_tokens, text

    def _labels2input(self, labels):
        inputs = [self.source_vocab.start_token]
        for field in self.FIELDS:
            value = labels.get(field, "N/A").replace(" ", "_")
            inputs.append(field.replace("_", "").upper() + "_" + value)
        if self.delex:
            if inputs[2] != "NEAR_N/A":
                inputs[2] = "NEAR_PRESENT"
            inputs.pop(-1)  
                
        inputs.append(self.source_vocab.stop_token)
        return inputs

    def _batch_labels(self, labels_batch):
        input_tokens = torch.LongTensor([
            [self.source_vocab[tok] for tok in self._labels2input(labels)]
            for labels in labels_batch
        ])
        length = input_tokens.size(1)
        inputs = Variable(
            input_tokens.t(),
            lengths=torch.LongTensor([length] * len(labels_batch)),
            length_dim=0, batch_dim=1, pad_value=-1) 
        if self._gpu > -1:
            inputs = inputs.cuda(self._gpu)
        return {"source_inputs": inputs}

    def _get_default_checkpoint(self, env):
        for ckpt, md in env["checkpoints"].items():
            if md.get("default", False):
                return ckpt
        return ckpt


