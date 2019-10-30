if [[ $NLG_GPU == "" ]]; then
    NLG_GPU=-1
fi


plumr generator_configs/e2e.aug_generator.scratch.train.clfmrs3-8.delex.jsonnet --proj aug_generators --run train --gpu $NLG_GPU

plumr generator_configs/e2e.aug_generator.scratch.train.rulemrs3-8.delex.jsonnet --proj aug_generators --run train --gpu $NLG_GPU

plumr generator_configs/e2e.aug_generator.scratch.train.rulemrs3-8.lex.jsonnet --proj aug_generators --run train --gpu $NLG_GPU

