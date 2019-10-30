import plum
from plum.types import register, PlumObject, HP, SM, props, Variable
import torch
import numpy as np
from itertools import combinations, product
from scipy.special import comb
import json
from d2t.postedit import e2e as postedit



@register("d2t.e2e_systematic_generation")
class E2ESystematicGeneration(PlumObject):
    
    FIELDS = ["eat_type", "near", "area", "family_friendly", 
              "customer_rating", "price_range", "food", "name"]

    FIELD_DICT = {

        "food": [
             'French',
             'Japanese',
             'Chinese',
             'English',
             'Indian',
             'Fast food',
             'Italian'
        ],
        "family_friendly": ['no', 'yes'],
        "area": ['city centre', 'riverside'],
        "near": [
             'Café Adriatic',
             'Café Sicilia',
             'Yippee Noodle Bar',
             'Café Brazil',
             'Raja Indian Cuisine',
             'Ranch',
             'Clare Hall',
             'The Bakers',
             'The Portland Arms',
             'The Sorrento',
             'All Bar One',
             'Avalon',
             'Crowne Plaza Hotel',
             'The Six Bells',
             'Rainbow Vegetarian Café',
             'Express by Holiday Inn',
             'The Rice Boat',
             'Burger King',
             'Café Rouge',
        ],
        "eat_type": ['coffee shop', 'pub', 'restaurant'],
        "customer_rating": ['3 out of 5', '5 out of 5', 'high', 
                            'average', 'low', '1 out of 5'],
        "price_range": ['more than £30', 'high', '£20-25', 'cheap', 
                        'less than £20', 'moderate'],
        "name": [
             'Cocum',
             'Midsummer House',
             'The Golden Curry',
             'The Vaults',
             'The Cricketers',
             'The Phoenix',
             'The Dumpling Tree',
             'Bibimbap House',
             'The Golden Palace',
             'Wildwood',
             'The Eagle',
             'Taste of Cambridge',
             'Clowns',
             'Strada',
             'The Mill',
             'The Waterman',
             'Green Man',
             'Browns Cambridge',
             'Cotto',
             'The Olive Grove',
             'Giraffe',
             'Zizzi',
             'Alimentum',
             'The Punter',
             'Aromi',
             'The Rice Boat',
             'Fitzbillies',
             'Loch Fyne',
             'The Cambridge Blue',
             'The Twenty Two',
             'Travellers Rest Beefeater',
             'Blue Spice',
             'The Plough',
             'The Wrestlers',
        ],


    }
    #batches = HP()
    checkpoint = HP(required=False)
    mr_size = HP(type=props.INTEGER, required=True)
    batch_size = HP(default=8, type=props.INTEGER)
    beam_size = HP(default=1, type=props.INTEGER)
    source_vocab = HP()
    target_vocab = HP()
    filename = HP()
    delex = HP(default=False, type=props.BOOLEAN)

    def _get_default_checkpoint(self, env):
        for ckpt, md in env["checkpoints"].items():
            if md.get("default", False):
                return ckpt
        return ckpt

    def _field_subsets_iter(self, size):

        for subset in combinations(self.FIELDS[:-1], size - 1):
            yield ("name",) + subset

    def _instance_iter(self, fields):
        options = [self.FIELD_DICT[f] for f in fields]
        for values in product(*options):
            yield {field: value for field, value in zip(fields, values)}

    @property
    def total_subsets(self):
        return int(comb(7, self.mr_size - 1))

    def total_settings(self, fields):
        return np.prod([len(self.FIELD_DICT[f]) for f in fields])

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

    def _get_outputs(self, model, labels):
        batch = self._batch_labels(labels)
        state = model.encode(batch)
        if self.beam_size > 1:
            search = plum.seq2seq.search.BeamSearch(
                max_steps=100, beam_size=self.beam_size,
               vocab=self.target_vocab)
        else:
            search = plum.seq2seq.search.GreedySearch(
                max_steps=100, vocab=self.target_vocab)
        search(model.decoder, state)
        outputs = search.output()

        raw_tokens = []
        postedited_outputs = []
        for i, output in enumerate(outputs):
            raw_tokens.append(output[:-1])
            output = postedit.detokenize(output)
            output = postedit.lexicalize(output, labels[i])
#            print(labels[i])
#            print(output)
#            input()
            postedited_outputs.append(output)

#        print()
        #input()

        return raw_tokens, postedited_outputs

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

        with output_path.open("w") as fp:

            field_subsets = self._field_subsets_iter(self.mr_size)
            for i, field_subset in enumerate(field_subsets, 1):
                print("slot subset {}/{}".format(i, self.total_subsets))
                if verbose:
                    print("    slots: {}".format(field_subset))

                total_mrs = self.total_settings(field_subset)

                batch = []
                inst_iter = self._instance_iter(field_subset)
                for j, labels in enumerate(inst_iter, 1):
                    print("setting {}/{}".format(j, total_mrs),
                          end="\r" if j < total_mrs else "\n", flush=True)
                    batch.append(labels)
                    if len(batch) == self.batch_size:
                        output_tokens, output_strings = self._get_outputs(
                            model, batch)
                        for labels, tokens, string in zip(
                                batch, output_tokens, output_strings):
                            data = json.dumps({
                                "labels": labels,
                                "tokens": tokens,
                                "text": string
                            })
                            print(data, file=fp)
                        batch = []           
                
                if len(batch) > 0:
                    output_tokens, output_strings = self._get_outputs(
                        model, batch)
                    for labels, tokens, string in zip(
                            batch, output_tokens, output_strings):
                        data = json.dumps({
                            "labels": labels,
                            "tokens": tokens,
                            "text": string
                        })
                        print(data, file=fp)
