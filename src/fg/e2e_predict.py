import plum
from plum.types import register, PlumObject, HP, SM, props, Variable
import torch
import numpy as np
from itertools import combinations, product
from scipy.special import comb
import json
from d2t.postedit import e2e as postedit

@register("d2t.e2e_predict")
class E2EPredict(PlumObject):
 
    checkpoint = HP(required=False)
    beam_size = HP(default=1, type=props.INTEGER)
    source_vocab = HP()
    target_vocab = HP()
    input_path = HP(type=props.EXISTING_PATH)
    filename = HP()
    delex = HP(default=False, type=props.BOOLEAN)

   
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
                labels = json.loads(line)["labels"]
                
                tokens, text = self._get_outputs(model, labels)
                data = json.dumps({
                    "labels": labels,
                    "tokens": tokens,
                    "text": text,
                })
                print(data, file=out_fp)

    def _get_outputs(self, model, labels):
        batch = self._batch_labels([labels])
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
        raw_tokens = outputs[0][:-1]
        postedited_output = postedit.detokenize(raw_tokens)
        postedited_output = postedit.lexicalize(postedited_output, labels)

        return raw_tokens, postedited_output

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


