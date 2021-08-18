# Run allennlp training locally

#
# edit these variables before running script
DATASET='chemprot'
TASK='text_classification'
with_finetuning='_finetune' #'_finetune'  # or '' for not fine tuning
dataset_size=4169

export BERT_VOCAB=/import/snvm-pa-scratch2/tonyk/scibert/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/import/snvm-pa-scratch2/tonyk/scibert/scibert_scivocab_uncased/weights.tar.gz

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning".json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=false
export TRAIN_PATH=data/$TASK/$DATASET/train.txt
export DEV_PATH=data/$TASK/$DATASET/dev.txt
export TEST_PATH=data/$TASK/$DATASET/test.txt

export CUDA_DEVICE=0

export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=2
export LEARNING_RATE=1e-5

python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s "$@"
