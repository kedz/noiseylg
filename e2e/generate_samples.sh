if [[ $NLG_GPU == "" ]]; then
    NLG_GPU=-1
fi

DELEX_MDL=base_generators/delex/train/run1/model_checkpoints/
DELEX_SRC_VOCAB=base_generators/delex/vocabs/source_vocab.pth
DELEX_TGT_VOCAB=base_generators/delex/vocabs/target_vocab.pth
DELEX_FAMILY_FRIENDLY_PATH="base_classifiers/delex/family_friendly/train/run1/model_checkpoints/"
DELEX_FAMILY_FRIENDLY_VOCAB_PATH="base_classifiers/delex/family_friendly/vocabs/family_friendly.pth"
DELEX_EAT_TYPE_PATH="base_classifiers/delex/eat_type/train/run1/model_checkpoints/"
DELEX_EAT_TYPE_VOCAB_PATH="base_classifiers/delex/eat_type/vocabs/eat_type.pth"
DELEX_PRICE_RANGE_PATH="base_classifiers/delex/price_range/train/run1/model_checkpoints/"
DELEX_PRICE_RANGE_VOCAB_PATH="base_classifiers/delex/price_range/vocabs/price_range.pth"
DELEX_CUSTOMER_RATING_PATH="base_classifiers/delex/customer_rating/train/run1/model_checkpoints/"
DELEX_CUSTOMER_RATING_VOCAB_PATH="base_classifiers/delex/customer_rating/vocabs/customer_rating.pth"
DELEX_FOOD_PATH="base_classifiers/delex/food/train/run1/model_checkpoints/"
DELEX_FOOD_VOCAB_PATH="base_classifiers/delex/food/vocabs/food.pth"
DELEX_AREA_PATH="base_classifiers/delex/area/train/run1/model_checkpoints/"
DELEX_AREA_VOCAB_PATH="base_classifiers/delex/area/vocabs/area.pth"

LEX_MDL=base_generators/lex/train/run1/model_checkpoints/ 
LEX_SRC_VOCAB=base_generators/lex/vocabs/source_vocab.pth
LEX_TGT_VOCAB=base_generators/lex/vocabs/target_vocab.pth

for sz in 3 4 5 6 7 8; do
    echo "GENERATING LEX SAMPLES WITH MR SIZE $sz AND RULE MR PARSER"
    python generate_samples_rule_lex.py \
        $LEX_MDL \
        $sz ../data/e2e/e2e.train.mrs.v6.lex.jsonl \
        $LEX_SRC_VOCAB $LEX_TGT_VOCAB \
        samples/base.lex.rules.mr$sz.max25k.jsonl \
        --max-mr-sample 25000 --gpu $NLG_GPU
done

for sz in 3 4 5 6 7 8; do
    echo "GENERATING DELEX SAMPLES WITH MR SIZE $sz AND RULE MR PARSER"
python generate_samples_rule_delex.py \
    $DELEX_MDL \
    $sz ../data/e2e/e2e.train.mrs.v5.delex.jsonl \
    $DELEX_SRC_VOCAB $DELEX_TGT_VOCAB \
    samples/base.delex.rules.mr$sz.max25k.jsonl \
    --max-mr-sample 25000 --gpu $NLG_GPU
done

for sz in 3 4 5 6 7 8; do
    echo "GENERATING DELEX SAMPLES WITH MR SIZE $sz AND CLASSIFIER MR PARSER"
python generate_samples_clf_delex.py \
    $DELEX_MDL \
    $sz ../data/e2e/e2e.train.mrs.v5.delex.jsonl \
    $DELEX_SRC_VOCAB $DELEX_TGT_VOCAB \
    samples/base.delex.clf.mr$sz.max25k.jsonl \
    --max-mr-sample 25000 --gpu $NLG_GPU \
    --area-ckpts $DELEX_AREA_PATH \
    --area-vocab $DELEX_AREA_VOCAB_PATH \
    --eat-type-ckpts $DELEX_EAT_TYPE_PATH \
    --eat-type-vocab $DELEX_EAT_TYPE_VOCAB_PATH \
    --food-ckpts $DELEX_FOOD_PATH \
    --food-vocab $DELEX_FOOD_VOCAB_PATH \
    --family-friendly-ckpts $DELEX_FAMILY_FRIENDLY_PATH \
    --family-friendly-vocab $DELEX_FAMILY_FRIENDLY_VOCAB_PATH \
    --price-range-ckpts $DELEX_PRICE_RANGE_PATH \
    --price-range-vocab $DELEX_PRICE_RANGE_VOCAB_PATH \
    --customer-rating-ckpts $DELEX_CUSTOMER_RATING_PATH \
    --customer-rating-vocab $DELEX_CUSTOMER_RATING_VOCAB_PATH 
done

