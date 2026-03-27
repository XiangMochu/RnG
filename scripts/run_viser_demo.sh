python \
viser_demo.py --config "configs/RnG_obj_medium_bf16_40k.yaml" \
 training.checkpoint_dir = 'experiments/checkpoints/RnGUP_Med' \
 training.val_dataset_cfgs.split_file = 'data/gso.txt' \
 training.batch_size_per_gpu = 1 \
 training.target_has_input =  false \
 training.num_input_views = 4 \
 inference.if_inference = true \
 inference.compute_metrics = true \
 inference.render_video = false \