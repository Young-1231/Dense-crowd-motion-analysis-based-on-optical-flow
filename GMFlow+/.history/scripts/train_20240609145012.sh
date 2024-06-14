# chairs
NUM_GPUS=1
CHECKPOINT_DIR=checkpoints_flow/chairs-gmflow-scale1 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage tub \
--tub_IM 1 \
--tub_root ~/autodl-tmp/TUBCrowdFlow \
--val_dataset tub \
--batch_size 1 \
--lr 4e-4 \
--image_size 720 1280 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 5000 \
--save_ckpt_freq 5000 \
--num_steps 5000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log