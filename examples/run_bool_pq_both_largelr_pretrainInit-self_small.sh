MODEL=t5-small
MAX_LENGTH=256
MAX_STEPS=30000
PREFIX_LENGTH=40
R=45
for TASK_NAME in superglue-boolq; do # cola mrpc mnli qnli qqp rte sst2 stsb
  for SCAP_LR in 1e-4 5e-4 1e-3 5e-3; do #  1e-4 5e-4 1e-3 5e-3
      for lr in 1 5 10; do #  3e-1 4e-1 5e-1
        for i in "32 17 256 30"; do # "8 7" "16 12" "32 20"
            set -- $i # Convert the "tuple" into the param args $1 $2...
                CUDA_VISIBLE_DEVICES=0 python train.py \
                    --pretrain_scpp_ckpt /path/to/prepended_prompt_ckpt \
                    --pretrain_scap_ckpt /path/to/added_prompt_ckpt \
                    --peft_type PROMPT_TUNING_LORA \
                    --pretrain_init True \
                    --scpp True \
                    --scap True \
                    --added_embedding_lr ${SCAP_LR} \
                    --learning_rate ${lr} \
                    --sub_dim_scpp $1 \
                    --codebook_size_scpp $2 \
                    --sub_dim_scap $3 \
                    --codebook_size_scap $4 \
                    --prefix_length ${PREFIX_LENGTH} \
                    --r ${R} \
                    --task_name ${TASK_NAME} \
                    --dataset_config_name en \
                    --model_name_or_path ${MODEL} \
                    --do_train \
                    --do_eval \
                    --do_predict \
                    --per_device_train_batch_size 16 \
                    --per_device_eval_batch_size 16 \
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
done;

