export CUDA_VISIBLE_DEVICES=0


python inference_test.py \
    --num_images 1 \
    --prompt "泰坦尼克号在还海上" \
    --model_path ./saved_model/LoRA_fusion_model \
    --save_path ./generated_images/ \
    --use_large_model $USE_LARGE_MODEL \
    --use_clip_similarity $USE_CLIP_SIMILARITY \
    #--compare_with_original



# python inference_test.py \
#     --num_images 1 \
#     --prompt "……" \
#     --model_path ./saved_model/LoRA_fusion_model \
#     --save_path ./generated_images/ \
#     --use_large_model $USE_LARGE_MODEL \
#     --use_clip_similarity $USE_CLIP_SIMILARITY

