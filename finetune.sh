### start pretrain
dataset_path="dataset/zinc/all.txt"
vocab_path="dataset/zinc/clique.txt"
input_model_path="saved_model/init"
output_model_path="saved_model/pretrain"
#data_path="./demo_zinc_smiles"
python finetune.py \
		--model=mgssl \
		--dataset=zinc \
		--gnn=gin \
		--epoch=500 \
		--batch_size=512 \
		--lr=1e-3 \
		--n_layers=4 \
		--emb_dim=64 \
		--decay=0
		--dropout=0.1 \
		--graphpooling=mean \
		--hidden_size=256 \
		--latent_size=128 \
		--JK=last \
		--dataset_path=$dataset_path \
		--vocab=$vocab_path \
		--order=bfs \
		--num_workers=4 \
		--seed=2022 \
		--input_model_file=$compound_encoder_config \
		--output_model_file=$model_config
