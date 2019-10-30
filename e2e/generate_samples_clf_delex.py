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
from plum.types import Variable


FIELDS = ["eat_type", "near", "area", "family_friendly", 
          "customer_rating", "price_range", "food"]
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
    "near": ["PRESENT"],
    "eat_type": ['coffee shop', 'pub', 'restaurant'],
    "customer_rating": ['3 out of 5', '5 out of 5', 'high', 
                        'average', 'low', '1 out of 5'],
    "price_range": ['more than £30', 'high', '£20-25', 'cheap', 
                    'less than £20', 'moderate'],
}


FAMILY_FRIENDLY_PATH = "../base_classifiers/delex/family_friendly/train/run1/model_checkpoints/model.ckpt.29.pth"
FAMILY_FRIENDLY_VOCAB_PATH = "../base_classifiers/delex/family_friendly/vocabs/family_friendly.pth"
EAT_TYPE_PATH = "../base_classifiers/delex/eat_type/train/run1/model_checkpoints/model.ckpt.2.pth"
EAT_TYPE_VOCAB_PATH = "../base_classifiers/delex/eat_type/vocabs/eat_type.pth"
PRICE_RANGE_PATH = "../base_classifiers/delex/price_range/train/run1/model_checkpoints/model.ckpt.30.pth"
PRICE_RANGE_VOCAB_PATH = "../base_classifiers/delex/price_range/vocabs/price_range.pth"
CUSTOMER_RATING_PATH = "../base_classifiers/delex/customer_rating/train/run1/model_checkpoints/model.ckpt.25.pth"
CUSTOMER_RATING_VOCAB_PATH = "../base_classifiers/delex/customer_rating/vocabs/customer_rating.pth"

FOOD_PATH = "../base_classifiers/delex/food/train/run1/model_checkpoints/model.ckpt.19.pth"
FOOD_VOCAB_PATH = "../base_classifiers/delex/food/vocabs/food.pth"
AREA_PATH = "../base_classifiers/delex/area/train/run1/model_checkpoints/model.ckpt.26.pth"
AREA_VOCAB_PATH = "../base_classifiers/delex/area/vocabs/area.pth"



def count_fields(path, size):
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with path.open("r") as fp:

        for line in fp:
            data = json.loads(line)      
            num_na = sum([field == "N/A" for field in data["labels"].values()])
            num_active = len(FIELDS) - num_na + 1

            for field, label in data["labels"].items():
                if field == "near" and label != "N/A":
                    label = "PRESENT"
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

    sample_fields = list(FIELDS)
    random.shuffle(sample_fields)
    sample_fields = sample_fields[:mr_size - 1]

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
        if field == "name":
            continue
        for label in FIELD_DICT[field]:
            if label == "N/A":
                continue
            c = label_counts[label]
            finished.append(c >= thresholds[field])
    return all(finished), sum(finished), len(finished)

def labels2input(labels, vocab):
    inputs = [vocab.start_token]
    for field in FIELDS:
        value = labels.get(field, "N/A").replace(" ", "_")
        inputs.append(field.replace("_", "").upper() + "_" + value)
    if inputs[2] != "NEAR_N/A":
        inputs[2] = "NEAR_PRESENT"
    inputs.pop(-1)  
            
    inputs.append(vocab.stop_token)
    return inputs

def make_generator_input(labels, vocab, gpu=-1):
    input_tokens = torch.LongTensor([
        [vocab[tok] for tok in labels2input(labels, vocab)]
    ])
    length = input_tokens.size(1)
    inputs = plum.types.Variable(
        input_tokens.t(),
        lengths=torch.LongTensor([length]),
        length_dim=0, batch_dim=1, pad_value=-1) 
    if gpu > -1:
        inputs = inputs.cuda(gpu)
    return {"source_inputs": inputs}


def rule_classify(text):
    pred_labels = {}
    for field in ["name", "near", "area", "eat_type", "food", "price_range",
                  "family_friendly", "customer_rating"]:

        val = rules.__dict__[field](text, delex=True)
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

def search2inputs(search_outputs, vocab, gpu):
    input_lens = [len(x) + 1 for x in search_outputs]
    max_len = max(input_lens)
    inputs = []
    for out in search_outputs:
        out = [vocab.start_token] + out
        if len(out) < max_len:
            out = out + [vocab.pad_token] * (max_len - len(out))
        inputs.append([vocab[t] for t in out])
    inputs = torch.LongTensor(inputs)
    input_lens = torch.LongTensor(input_lens)
    inputs = Variable(
        inputs.t(),
        lengths=input_lens,
        batch_dim=1, length_dim=0, pad_value=vocab.pad_index)
    if gpu > -1:
        inputs = inputs.cuda(gpu)
    return {"inputs": inputs}

def filter_candidates(descs, clf_input, clfs, label_vocabs):

    clf_states = {field: clf(clf_input) for field, clf in clfs.items()}

    filtered = []
    for i, descs in enumerate(descs):

        if "__NAME__" not in descs:
            continue
        mr = {"name": "__NAME__"}

        if "__NEAR__" in descs:
            mr["near"] = "PRESENT"
        else:
            mr["near"] = "N/A"

        probs = []
        for field, state in clf_states.items():
            prob, index = state["probs"][i].max(0)
            probs.append(prob.item())
            label = label_vocabs[field][index.item()]
            mr[field] = label

        if any([p < .5 for p in probs]):
            continue

        filtered.append([descs[:-1], mr])

    return filtered

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
    parser.add_argument("--area-ckpts", type=Path)
    parser.add_argument("--area-vocab", type=Path)
    parser.add_argument("--eat-type-ckpts", type=Path)
    parser.add_argument("--eat-type-vocab", type=Path)
    parser.add_argument("--food-ckpts", type=Path)
    parser.add_argument("--food-vocab", type=Path)
    parser.add_argument("--family-friendly-ckpts", type=Path)
    parser.add_argument("--family-friendly-vocab", type=Path)
    parser.add_argument("--price-range-ckpts", type=Path)
    parser.add_argument("--price-range-vocab", type=Path)
    parser.add_argument("--customer-rating-ckpts", type=Path)
    parser.add_argument("--customer-rating-vocab", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(exist_ok=True, parents=True)

    clfs = {
        "area": load_model(args.area_ckpts),  
        "eat_type": load_model(args.eat_type_ckpts),
        "food": load_model(args.food_ckpts),
        "family_friendly": load_model(args.family_friendly_ckpts),
        "price_range": load_model(args.price_range_ckpts),
        "customer_rating": load_model(args.customer_rating_ckpts),
    }
    
    label_vocabs = {
        "area": plum.load(args.area_vocab),
        "eat_type": plum.load(args.eat_type_vocab),
        "food": plum.load(args.food_vocab),
        "family_friendly": plum.load(args.family_friendly_vocab),
        "price_range": plum.load(args.price_range_vocab),
        "customer_rating": plum.load(args.customer_rating_vocab),
    }

    generator = load_model(args.model)
    if args.gpu > -1:
        generator.cuda(args.gpu)
        for m in clfs.values():
            m.cuda(args.gpu)
    src_vocab = plum.load(args.src_vocab)
    tgt_vocab = plum.load(args.tgt_vocab)
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
                                              gpu=args.gpu)
            encoder_state = generator.encode(gen_inputs)
            search = plum.seq2seq.search.GreedyNPAD(
                max_steps=100, std=1.0, samples=200, vocab=tgt_vocab)
            search(generator.decoder, encoder_state)
            candidate_descriptions = search.output(n_best=20)[0]

            clf_input = search2inputs(candidate_descriptions, tgt_vocab,
                                      args.gpu)
            candidates = filter_candidates(candidate_descriptions, clf_input,
                                           clfs, label_vocabs)

            for desc, mr in candidates:
                pretty = postedit.detokenize(desc)

                if pretty in found:
                    total_rejected += 1
                    continue
                found.add(pretty)
                 
                data = {
                    "sequence": {
                        "mrs": labels2inputs(mr),
                        "tokens": desc[:-1]
                    },
                    "labels": mr, 
                }
                print(json.dumps(data), file=fp)

                total_accepted += 1
                pred_mr_size = count_active_fields(mr)
                
                for field, value in mr.items():
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
