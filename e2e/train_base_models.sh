if [[ $NLG_GPU == "" ]]; then
    NLG_GPU=-1
fi

plumr generator_configs/e2e.base_generator.delex.jsonnet --proj base_generators/delex --run train --gpu $NLG_GPU
plumr generator_configs/e2e.base_generator.lex.jsonnet --proj base_generators/lex  --run train --gpu $NLG_GPU
