
To run the experiments in the paper, i.e. to obtain the base and augmented
generators, 4 individual steps must be run:

1. Training Base Generators
2. Training Classifier MR Parser
3. Generating Samples
4. Training Augmented Generators

Please see the individual instructions for each step below. Alternatively,
you can download the results of each step that were used in paper.
See individual sections for download links.

Downloading All Pretrained Models/Data
======================================

You can download all pretrained models/sample outputs by running:

>> bash download_paper_data.sh

Generating From Training Models
===============================

To see a code example of how to generate text using greedy decoding, 
beam decoding, ancestral sampling, and noise injection sampling please
see the bash script example_generation.sh and python script 
example_generation.py.

Reproducing Automatic Metrics
=============================

After running steps 1-4 above, run the evaluation script:

>> bash evaluate_generators.sh

Test outputs can be found in the jsonl file in:

base_generators/delex/generate_test_greedy/run1/output/base.delex.greedy.test.jsonl
base_generators/delex/generate_test_greedy/run1/output/base.delex.beam8.test.jsonl

base_generators/lex/generate_test_greedy/run1/output/base.lex.greedy.test.jsonl
base_generators/lex/generate_test_greedy/run1/output/base.lex.beam8.test.jsonl

aug_generators/delex/scratch.train.rulemrs3-8/generate_test_beam8/run1/output/aug.rule.delex.greedy.test.jsonl
aug_generators/delex/scratch.train.rulemrs3-8/generate_test_beam8/run1/output/aug.rule.delex.beam8.test.jsonl

aug_generators/delex/scratch.train.clfmrs3-8/generate_test_beam8/run1/output/aug.clf.delex.greedy.test.jsonl
aug_generators/delex/scratch.train.clfmrs3-8/generate_test_beam8/run1/output/aug.clf.delex.beam8.test.jsonl

aug_generators/lex/scratch.train.rulemrs3-8/generate_test_beam8/run1/output/aug.rule.greedy.test.jsonl
aug_generators/lex/scratch.train.rulemrs3-8/generate_test_beam8/run1/output/aug.rule.beam8.test.jsonl

Each line corresponds to an output for the corresponding line in the E2E test
set.


Using a GPU
===========

These scripts require use of a gpu to train in a practical amount of time.
To set the experiments to run on gpu 0, run:

>> export NLG_GPU=0


1. Training Base Generators
===========================

To train the base generators described in the paper run:

>> bash train_base_models.sh

After the training has completed, you can find the delexicalized models
in 

  base_generators/delex/train/run1/model_checkpoints

There will be multiple checkpoints in this directory.
The best model is specified in the ckpt.metadata.json, in the
"optimal_checkpoint" field.

The lexicalized base models can be found analogously in

  base_generators/lex/train/run1/model_checkpoints

To download the base generators used in the paper, download this link:
    https://drive.google.com/file/d/1aLdcaO9J99UulzR9wozBh9d1LV878oLm/view?usp=sharing


2. Training Classifier MR Parser
================================

To train the CNN classifiers for the classifier based MR Parser, run:

>> bash train_classifiers.sh

After the training has completed, you can find the classifier models
in 

  base_classifiers/delex/ATTRIBUTE/train/run1/model_checkpoints

where ATTRIBUTE is one of area, eat_type, etc. 
There will be multiple checkpoints in this directory.
The best model is specified in the ckpt.metadata.json, in the
"optimal_checkpoint" field.

To download the classifiers used in the paper, download this link:
    https://drive.google.com/file/d/1XTrVdVQ_vKrXEt83qf0m3NfVedRf1ifb/view?usp=sharing


3. Generate Samples
===================

To generate the samples using the base models and MR parsers run:
 
>> bash generate_samples.sh

Generated samples can be found in the samples directory.

To download the samples used in the paper, download this link:
    https://drive.google.com/file/d/1qb-5BvlOTqstXn2yyNJwT7AqlUj2bkC2/view?usp=sharing


4. Train Augmented Generators
============================= 

To train the base generators described in the paper run:

>> bash train_aug_models.sh

After the training has completed, you can find the finished augmented 
generators in 

  aug_generators/delex/scratch.train.rulemrs3-8/train/run1/model_checkpoints
  aug_generators/delex/scratch.train.clfmrs3-8/train/run1/model_checkpoints
  aug_generators/lex/scratch.train.rulemrs3-8/train/run1/model_checkpoints

There will be multiple checkpoints in these directory.
The best model is specified in the ckpt.metadata.json, in the
"optimal_checkpoint" field.

To download the augmented generators used in the paper, download this link:
    https://drive.google.com/file/d/1hQpRR3VC1w8gNjlViOuLZGAukEHkR6Ij/view?usp=sharing


