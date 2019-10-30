if [[ $NLG_GPU == "" ]]; then
    NLG_GPU=-1
fi
TESTSET="../data/e2e/e2e.test.refs.txt"

echo "Base Delex Greedy"
# Generate outputs for test set.
plumr generator_configs/e2e.base_generator.delex.jsonnet --proj base_generators/delex --run generate_test_greedy --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py base_generators/delex/generate_test_greedy/run1/output/base.delex.greedy.test.jsonl $TESTSET

echo "Base Delex Beam"
# Generate outputs for test set.
plumr generator_configs/e2e.base_generator.delex.jsonnet --proj base_generators/delex --run generate_test_beam8 --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py base_generators/delex/generate_test_beam8/run1/output/base.delex.beam8.test.jsonl $TESTSET


echo "Delex Aug Gen. Rule MR Parser Greedy"
# Generate outputs for test set.
plumr generator_configs/e2e.aug_generator.scratch.train.rulemrs3-8.delex.jsonnet --proj aug_generators/delex/scratch.train.rulemrs3-8 --run generate_test_greedy --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py aug_generators/delex/scratch.train.rulemrs3-8/generate_test_greedy/run1/output/aug.rule.delex.greedy.test.jsonl $TESTSET

echo "Lex Aug Gen. Rule MR Parser Beam"
# Generate outputs for test set.
plumr generator_configs/e2e.aug_generator.scratch.train.rulemrs3-8.delex.jsonnet --proj aug_generators/delex/scratch.train.rulemrs3-8 --run generate_test_beam8 --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py aug_generators/delex/scratch.train.rulemrs3-8/generate_test_beam8/run1/output/aug.rule.delex.beam8.test.jsonl $TESTSET

echo "Delex Aug Gen. Classifier MR Parser Greedy"
# Generate outputs for test set.
plumr generator_configs/e2e.aug_generator.scratch.train.clfmrs3-8.delex.jsonnet --proj aug_generators/delex/scratch.train.clfmrs3-8 --run generate_test_greedy --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py aug_generators/delex/scratch.train.clfmrs3-8/generate_test_greedy/run1/output/aug.clf.delex.greedy.test.jsonl $TESTSET

echo "Lex Aug Gen. Classifier MR Parser Beam"
# Generate outputs for test set.
plumr generator_configs/e2e.aug_generator.scratch.train.clfmrs3-8.delex.jsonnet --proj aug_generators/delex/scratch.train.clfmrs3-8 --run generate_test_beam8 --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py aug_generators/delex/scratch.train.clfmrs3-8/generate_test_beam8/run1/output/aug.clf.delex.beam8.test.jsonl $TESTSET


echo "Base Lex Greedy"
# Generate outputs for test set.
plumr generator_configs/e2e.base_generator.lex.jsonnet --proj base_generators/lex --run generate_test_greedy --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py base_generators/lex/generate_test_greedy/run1/output/base.lex.greedy.test.jsonl $TESTSET

echo "Base Lex Beam"
# Generate outputs for test set.
plumr generator_configs/e2e.base_generator.lex.jsonnet --proj base_generators/lex --run generate_test_beam8 --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py base_generators/lex/generate_test_beam8/run1/output/base.lex.beam8.test.jsonl $TESTSET


echo "Lex Aug Gen. Rule MR Parser Greedy"
# Generate outputs for test set.
plumr generator_configs/e2e.aug_generator.scratch.train.rulemrs3-8.lex.jsonnet --proj aug_generators/lex/scratch.train.rulemrs3-8 --run generate_test_greedy --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py aug_generators/lex/scratch.train.rulemrs3-8/generate_test_greedy/run1/output/aug.rule.lex.greedy.test.jsonl $TESTSET

echo "Lex Aug Gen. Rule MR Parser Beam"
# Generate outputs for test set.
plumr generator_configs/e2e.aug_generator.scratch.train.rulemrs3-8.lex.jsonnet --proj aug_generators/lex/scratch.train.rulemrs3-8 --run generate_test_beam8 --gpu $NLG_GPU
# Run official eval scripts
python eval_model_test.py aug_generators/lex/scratch.train.rulemrs3-8/generate_test_beam8/run1/output/aug.rule.beam8.test.jsonl $TESTSET


