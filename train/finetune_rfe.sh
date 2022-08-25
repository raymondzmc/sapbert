CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--model_dir "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" \
	--train_dir  "../training_data/training_file_rfe_pairwise_pair_th50.txt" \
	--output_dir tmp/sapbert_rfe\
	--use_cuda \
	--epoch 1 \
	--train_batch_size 32 \
	--learning_rate 2e-5 \
	--max_length 25 \
	--checkpoint_step 999999 \
	--amp \
	--pairwise \
	--random_seed 33 \
	--loss ms_loss \
	--use_miner \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "cls"

CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
	--train_dir  "../training_data/training_file_rfe_pairwise_pair_th50.txt" \
	--output_dir tmp/pubmedbert_rfe\
	--use_cuda \
	--epoch 1 \
	--train_batch_size 32 \
	--learning_rate 2e-5 \
	--max_length 25 \
	--checkpoint_step 999999 \
	--amp \
	--pairwise \
	--random_seed 33 \
	--loss ms_loss \
	--use_miner \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "cls"
