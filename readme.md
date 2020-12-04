# BERT based models for Qustion Answering task


## About the project
we implemented 3 different BERT related models for this Question Answering task and the best models obtained 67.69\% test accuracy on the RACE dataset. In the experiment, we try to tune different parameters which include learning rate, L2 regulation, and batch size, and max sequence length. Moreover, we implement the Data augmentation on the ALBERT to obtain the 0.6 gain in accuracy. 

## About the code
- data_augmentation.py: use some simple way to do data augmentation in the RACE dataset.
- utils_multiple_choice.py: preprocess the data from RACE and others
- run_multiple_choice.py: load the pre-training model and run the training
- transformers/ : some setting from transformers
## some important reference:
- [transformers: A useful pre-training model framework for nlp](https://github.com/huggingface/transformers)
- [nltk: some tools for nlp](https://github.com/nltk/nltk)

## Usage
* train
 ```sh
CUDA_VISIBLE_DEVICES=0 nohup python run_multiple_choice.py \
--task_name semEval \
--model_name_or_path bert-base-uncased \
--data_dir "RACE/" \
--output_dir "out" \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--cache_dir "cache_dir"\
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps 2 \
--overwrite_output >bert.log 2>&1 &
  ```
* eval
```sh
python run_multiple_choice.py \
--task_name RACE \
--model_name_or_path "out/" \
--data_dir "RACE/" \
--output_dir "eval_out" \
--do_eval \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps 2 
```

## hardward resource
GeForce RTX 2080(11G) * 2

## Contribution
- Wu hao
- Yang siting
- Chan Chunkit