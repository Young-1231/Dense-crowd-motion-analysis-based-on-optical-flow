# 训练tub数据集
# --resume：加载权重路径
# --stage：训练数据集
# --tub_IM：用于训练的TUB场景
# --val_dataset：验证数据集
# --batch_size：批大小
# --lr：学习率
# --image_size：图像大小
# --padding_factor：填充因子
# --upsample_factor：上采样因子
# --with_speed_metric：是否使用速度度量
# --val_freq：验证频率
# --save_ckpt_freq：保存模型频率
# --num_steps：训练步数
NUM_GPUS=1
CHECKPOINT_DIR=checkpoints_flow/chairs-gmflow-scale1 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume pretrained/gmflow-scale1-things-e9887eda.pth \
--stage tub \
--tub_IM 1234 \
--tub_root ~/autodl-tmp/TUBCrowdFlow \
--val_dataset tub \
--batch_size 1 \
--lr 4e-4 \
--image_size 720 1280 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 2000 \
--save_ckpt_freq 2000 \
--num_steps 2000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log