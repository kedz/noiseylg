if [[ $NLG_GPU == "" ]]; then
    NLG_GPU=-1
fi

plumr classifier_configs/e2e.classifier.area.delex.v1.jsonnet --proj base_classifiers/area --run train --gpu $NLG_GPU
plumr classifier_configs/e2e.classifier.customer_rating.delex.v1.jsonnet --proj base_classifiers/customer_rating --run train --gpu $NLG_GPU
plumr classifier_configs/e2e.classifier.eat_type.delex.v1.jsonnet --proj base_classifiers/eat_type --run train --gpu $NLG_GPU
plumr classifier_configs/e2e.classifier.family_friendly.delex.v1.jsonnet --proj base_classifiers/family_friendly --run train --gpu $NLG_GPU
plumr classifier_configs/e2e.classifier.food.delex.v1.jsonnet --proj base_classifiers/food --run train --gpu $NLG_GPU
plumr classifier_configs/e2e.classifier.price_range.delex.v1.jsonnet --proj base_classifiers_v2/price_range --run train --gpu $NLG_GPU

