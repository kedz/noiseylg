import plum
from plum.types import register, PlumObject, HP, SM, props, Variable
import torch
import numpy as np
from itertools import combinations, product
from scipy.special import comb
import json
from d2t.preprocessing import laptops as preproc

VALUES = ["batteryrating", "driverange", "family",  "pricerange", 
		  "weightrange", "isforbusinesscomputing"]

@register("d2t.laptop_systematic_inform_count")
class LaptopSystematicInformCount(PlumObject):
 
    checkpoint = HP(required=False)
    beam_size = HP(default=1, type=props.INTEGER)
    source_vocab = HP()
    target_vocab = HP()
    filename = HP()

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

        samples = self.make_samples()

        with open(output_path, "w") as out_fp:
            for i, mr in enumerate(samples, 1):
                print("{}/{}".format(i, len(samples)), 
                      end="\r" if i < len(samples) else "\n", flush=True)
                
                gen_input = self.make_generator_inputs(mr)

                source = preproc.mr2source_inputs(mr)
                tokens = self._get_outputs(model, gen_input)
                data = json.dumps({
                    "source": source,
                    "mr": mr,
                    "text": " ".join(tokens),
                })
                print(data, file=out_fp, flush=True)


    def make_samples(self):
        
        settings = []
        for size in [1, 2, 3]:
            for fields in combinations(VALUES, size):
                fields = ("count",) + fields
                if 'isforbusinesscomputing' in fields:
                    idx = fields.index('isforbusinesscomputing')
                    for val in ['_true', '_false']:
                        out = list(fields)
                        out[idx] = out[idx] + val
                        settings.append(out)
                else:
                    
                    settings.append(fields)

        mrs = []
        for setting in settings:
            for p in product(*([['']] + [['', "_dontcare"]] * len(setting[:-1]))):
                if 'isforbusinesscomputing_false' in setting:
                    idx = setting.index('isforbusinesscomputing_false')
                    if p[idx] == "_dontcare":
                        continue
                mr = {"da": "inform_count", "fields": {}}
                for ignore, field in zip(p, setting): 
                    if field.startswith("isforbusinesscomputing"):
                        f, v = field.split("_")
                        if ignore == "_dontcare":
                            mr["fields"][f] = {"no_lex_value": "dontcare"}
                        else: 
                            mr["fields"][f] = {"no_lex_value": v}
                    else:
                        if ignore == "_dontcare":
                            mr["fields"][field] = {"no_lex_value": "dontcare"}
                        else:
                            mr["fields"][field] = {"lex_value": "PLACEHOLDER"}
                mrs.append(mr)
        return mrs

    def make_generator_inputs(self, data):
        source = preproc.mr2source_inputs(data)
        tokens = [self.source_vocab.start_token] + source \
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
        if self.beam_size > 1:
            search = plum.seq2seq.search.BeamSearch(
                max_steps=100, beam_size=self.beam_size,
               vocab=self.target_vocab)
        else:
            search = plum.seq2seq.search.GreedySearch(
                max_steps=100, vocab=self.target_vocab)
        search(model.decoder, state)
        outputs = search.output()
        raw_tokens = outputs[0][:-1]
        return raw_tokens

    def _get_default_checkpoint(self, env):
        for ckpt, md in env["checkpoints"].items():
            if md.get("default", False):
                return ckpt
        return ckpt


