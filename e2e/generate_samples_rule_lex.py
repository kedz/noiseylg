import argparse
from pathlib import Path
from collections import defaultdict
import json
import random
import numpy as np
import plum
import torch
from d2t.postedit import e2e as postedit
from d2t.rule_based_classifiers import e2e_v2 as rules


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


def count_fields(path, size):
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with path.open("r") as fp:

        for line in fp:
            data = json.loads(line)      
            num_na = sum([field == "N/A" for field in data["labels"].values()])
            num_active = len(FIELDS) - num_na

            for field, label in data["labels"].items():
                counts[num_active][field][label] += 1

    thresholds = defaultdict(lambda: defaultdict(int))
    for active in range(size, size + 1):
        print(active)
        for field, label_counts in counts[active].items():
            print("  ", field)
            thresholds[active][field] = max(label_counts.values())
            for label, count in label_counts.items():
                print("      ", label, count, 'th=',thresholds[active][field])
    return counts, thresholds

def print_counts(counts, thresholds, size):
    print("mr-size=", size)
    for field, label_counts in counts[size].items():
        print("  ", field)
        for label, count in label_counts.items():
            print("      ", label, count, "th=", thresholds[size][field])


def draw_samples(counts, mr_size):
    sample = {}

    sample_fields = list(FIELDS[:-1])
    random.shuffle(sample_fields)
    sample_fields = ["name"] + sample_fields[:mr_size - 1]

    for field in sample_fields:
        choices = []
        invcounts = []
        for label in FIELD_DICT[field]:
            if label == "N/A":
                continue
            choices.append(label)
            invcounts.append(1 / (counts[field][label] + 1))
        invcounts = np.array(invcounts)
        choice = np.random.choice(choices, p=invcounts/ np.sum(invcounts))
        sample[field] = choice
    return sample

def count_active_fields(labels):
    return sum([val != 'N/A' for val in labels.values()])

def check_terminate(counts, thresholds):

    finished = []

    for field, label_counts in counts.items():
        for label in FIELD_DICT[field]:
            if label == "N/A":
                continue
            c = label_counts[label]
            finished.append(c >= thresholds[field])
    return all(finished), sum(finished), len(finished)

def labels2input(labels, vocab, delex=False):
    inputs = [vocab.start_token]
    for field in FIELDS:
        value = labels.get(field, "N/A").replace(" ", "_")
        inputs.append(field.replace("_", "").upper() + "_" + value)
    if delex:
        if inputs[2] != "NEAR_N/A":
            inputs[2] = "NEAR_PRESENT"
        inputs.pop(-1)  
            
    inputs.append(vocab.stop_token)
    return inputs

def make_generator_input(labels, vocab, delex=False, gpu=-1):
    input_tokens = torch.LongTensor([
        [vocab[tok] for tok in labels2input(labels, vocab, delex)]
    ])
    length = input_tokens.size(1)
    inputs = plum.types.Variable(
        input_tokens.t(),
        lengths=torch.LongTensor([length]),
        length_dim=0, batch_dim=1, pad_value=-1) 
#    if self._gpu > -1:
#        inputs = inputs.cuda(self._gpu)
    if gpu > -1:
        inputs = inputs.cuda(gpu)
    return {"source_inputs": inputs}


def rule_classify(text):
    pred_labels = {}
    for field in ["name", "near", "area", "eat_type", "food", "price_range",
                  "family_friendly", "customer_rating"]:

        val = rules.__dict__[field](text)
        #vals = [rules.__dict__[field](text, cand_val) 
        #        for cand_val in FIELD_DICT[field]]
        #vals = [v for v in vals if v != "N/A"]
        #if len(vals) == 1:
        pred_labels[field] = val
    return pred_labels

def labels2inputs(labels):
    inputs = []
    for field in FIELDS:
        value = labels.get(field, "N/A").replace(" ", "_")
        inputs.append(field.replace("_", "").upper() + "_" + value)
    return inputs

def load_model(ckpt_dir):
    meta = json.loads((ckpt_dir / "ckpt.metadata.json").read_text())
    model_path = ckpt_dir / meta['optimal_checkpoint']
    return plum.load(model_path).eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("size", type=int)
    parser.add_argument("train_mrs", type=Path)
    parser.add_argument("src_vocab", type=Path)
    parser.add_argument("tgt_vocab", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-mr-sample", type=int, default=10000)
    parser.add_argument("--delex", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1)

    args = parser.parse_args()

    args.output.parent.mkdir(exist_ok=True, parents=True)

    generator = load_model(args.model)
    if args.gpu > -1:
        generator.cuda(args.gpu)
    src_vocab = plum.load(args.src_vocab)
    tgt_vocab = plum.load(args.tgt_vocab)
    print(src_vocab)
    print(tgt_vocab)
    counts, thresholds = count_fields(args.train_mrs, args.size)

    total_accepted = 0
    total_rejected = 0
    num_mr_samples = 0
    finish_round = False
    found = set()

    with args.output.open("w") as fp:
        while num_mr_samples < args.max_mr_sample and not finish_round:
            num_mr_samples += 1
            label_sample = draw_samples(counts[args.size], args.size)

            gen_inputs = make_generator_input(label_sample, src_vocab, 
                                              args.delex,
                                              gpu=args.gpu)
            encoder_state = generator.encode(gen_inputs)
            search = plum.seq2seq.search.GreedyNPAD(
                max_steps=100, std=1.0, samples=200, vocab=tgt_vocab)
            search(generator.decoder, encoder_state)
            candidate_descriptions = search.output(n_best=20)[0]

            for desc in candidate_descriptions:
                pretty = postedit.detokenize(desc)
                if pretty in found:
                    total_rejected += 1
                    continue
                if pretty.count("' ' '") > 0:
                    total_rejected += 1
                    continue
                pred_labels = rule_classify(pretty)
                if pred_labels["name"] is None:
                    total_rejected += 1
                    continue
                found.add(pretty)
                 
                data = {
                    "sequence": {
                        "mrs": labels2inputs(pred_labels),
                        "tokens": desc[:-1]
                    },
                    "labels": pred_labels,
                }
                print(json.dumps(data), file=fp)

                #print(label_sample)
                #print(pretty)
                #print({f: v for f, v in rule_classify(pretty).items()
                #       if v != "N/A"})
                #print()
                total_accepted += 1
                pred_mr_size = count_active_fields(pred_labels)
                

                for field, value in pred_labels.items():
                    #if value == "N/A":
#                    continue
                    counts[pred_mr_size][field][value] += 1

            finish_round, num_complete, num_total = check_terminate(
                counts[args.size], thresholds[args.size])
             
            print("{} / {}  accept={} reject={} samples-this-round={}".format(
                    num_complete, num_total, total_accepted, total_rejected,
                    num_mr_samples),
                end="\r" if not finish_round else "\n", flush=True)
    print()

    print_counts(counts, thresholds, args.size)

if __name__ == "__main__":
    main()
