## Deep learning models for Qustion Answering task
we implemented 3 different BERT related models for this Question Answering task and the best models obtained 67.69\% test accuracy on the RACE dataset. In the experiment, we try to tune different parameters which include learning rate, L2 regulation, and batch size, and max sequence length. Moreover, we implement the Data augmentation on the ALBERT to obtain the 0.6 gain in accuracy. 

### some important reference:
- transformers: A useful pre-training model framework for nlp
- nltk: some tools for nlp

### train
CUDA_VISIBLE_DEVICES=1 python run_multiple_choice.py \
--task_name RACE \
--model_name_or_path roberta-xxlarge \
--data_dir "sem_data/" \
--output_dir "out3" \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--cache_dir "test"\
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps 2 \
--overwrite_output

### eval
python run_multiple_choice.py \
--task_name RACE \
--model_name_or_path "out/" \
--data_dir "sem_data/" \
--output_dir "eval_out" \
--do_eval \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps 2 
