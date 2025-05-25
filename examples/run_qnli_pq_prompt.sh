MODEL=t5-base
MAX_LENGTH=256
MAX_STEPS=300000
PREFIX_LENGTH=60
R=30
for TASK_NAME in qnli; do # cola mrpc mnli qnli qqp rte sst2 stsb
  for LORA_LR in 1e-3 5e-3; do #  1e-4 5e-4 1e-3 5e-3
      for lr in 3e-1 4e-1 5e-1; do #  3e-1 4e-1 5e-1
        for i in "32 20"; do 
            set -- $i # Convert the "tuple" into the param args $1 $2...
                CUDA_VISIBLE_DEVICES=0 python train.py \
                    --peft_type PROMPT_TUNING_LORA \
                    --pq_prompt True \
                    --pq_lora False \
                    --lora_embedding_lr ${LORA_LR} \
                    --learning_rate ${lr} \
                    --sub_dim_prompt $1 \
                    --codebook_size_prompt $2 \
                    --prefix_length ${PREFIX_LENGTH} \
                    --r ${R} \
                    --task_name ${TASK_NAME} \
                    --dataset_config_name en \
                    --model_name_or_path ${MODEL} \
                    --do_train \
                    --do_eval \
                    --do_predict \
                    --per_device_train_batch_size 32 \
                    --per_device_eval_batch_size 32 \
                    --max_seq_length ${MAX_LENGTH} \
                    --save_strategy steps \
                    --evaluation_strategy steps \
                    --max_steps ${MAX_STEPS} \
                    --eval_steps 1000 \
                    --save_steps 1000 \
                    --warmup_steps 1800 \
                    --weight_decay 1e-2 \
                    --load_best_model_at_end \
                    --save_total_limit 1 \
                    --output_dir /path/to/output_dir
            done;
        done;
    done;
done