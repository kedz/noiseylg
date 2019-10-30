import argparse
from pathlib import Path
import json
import torch
import plum
from plum.types import Variable
from plum.seq2seq.search import (AncestralSampler, GreedyNPAD, GreedySearch, 
                                 BeamSearch)


def seq2tsr(seq, vocab):
    length = torch.LongTensor([len(seq)])
    tensor = torch.LongTensor([[vocab[x] for x in seq]]).t()
    return Variable(tensor, lengths=length, length_dim=0, batch_dim=1,
                    pad_value=vocab.pad_index)

def load_model(ckpt_dir):
    meta = json.loads((ckpt_dir / "ckpt.metadata.json").read_text())
    model_path = ckpt_dir / meta['optimal_checkpoint']
    return plum.load(model_path).eval()

def main():
    parser = argparse.ArgumentParser("Generation demo.")
    parser.add_argument("src_vocab", type=Path)
    parser.add_argument("tgt_vocab", type=Path)
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()

    source_vocab = plum.load(args.src_vocab)
    target_vocab = plum.load(args.tgt_vocab)
    model = load_model(args.checkpoint_dir)

    input_example = [
        "<sos>",
        "EATTYPE_N/A", 
        "NEAR_Crowne_Plaza_Hotel", 
        "AREA_city_centre", 
        "FAMILYFRIENDLY_no", 
        "CUSTOMERRATING_N/A", 
        "PRICERANGE_N/A", 
        "FOOD_N/A", 
        "NAME_Browns_Cambridge",
        "<eos>"    
    ]
   
    encoder_inputs = {"source_inputs": seq2tsr(input_example, source_vocab)}
    encoder_state = model.encode(encoder_inputs)

    print("Greedy Decoding")
    print("===============")
    print()
    greedy_search = GreedySearch(max_steps=100, vocab=target_vocab)
    greedy_search(model.decoder, encoder_state)
    for out in greedy_search.output():
        print(" ".join(out))
    print()

    print("Beam Decoding (beam size=8, showing top 3)")
    print("==========================================")
    print()
    beam_search = BeamSearch(max_steps=100, vocab=target_vocab, beam_size=10)
    beam_search(model.decoder, encoder_state)
    for out in beam_search.output(n_best=3)[0]:
        print(" ".join(out))
    print()

    print("Ancestral Sampling (samples=10, showing top 3)")
    print("==============================================")
    print()
    ancestral_sampler = AncestralSampler(max_steps=100, vocab=target_vocab, 
                                   samples=10)
    ancestral_sampler(model.decoder, encoder_state)
    for out in ancestral_sampler.output(n_best=3)[0]:
        print(" ".join(out))
    print()


    print("Noise Injection Sampling (samples=10, showing top 3)")
    print("====================================================")
    print()
    npad_search = GreedyNPAD(max_steps=100, vocab=target_vocab, samples=10, 
                             std=1.0)
    npad_search(model.decoder, encoder_state)
    for out in npad_search.output(n_best=3)[0]:
        print(" ".join(out))
    print()

if __name__ == "__main__":
    main()
