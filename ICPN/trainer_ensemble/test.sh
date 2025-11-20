python train_fsl.py  --max_epoch 20 --model_class ProtoNet  --backbone_class Res12 --use_euclidean --fix_BN  --shot 5 --eval_shot 5  --model_name PN_Ind  --step_size 10 --lr 0.0001   --temperature 20   --init_weights /home/liuziyi/cub/checkpoints/baseclass/MiniImageNet-ProtoNet-Res12-05w05s15q-Pre-DIS/PN_base_10_0.5_lr1e-05mul10_step_b0.1_bsz100_T120.0T232_5w_5s/min_loss.pth --gpu 0
# Cooperation
python train_fsl.py  --max_epoch 15  --model_class ProtoNet  --backbone_class Res12 --use_euclidean --fix_BN  --shot 5 --eval_shot 5  --model_name PN_Coo  --step_size 10 --lr 0.00001 --temperature 20   --trained_models /home/liuziyi/cub/trainer_ensemble/checkpoints/subsets/MiniImageNet-ProtoNet-Res12-05w05s15q-Pre-DIS/Ensemble:PN_Ind_0.5_10_wd:0.0005_lr0.0001_T120.0T232_5w_5s-FBN/min_loss.pth --distill --gpu 0
# Evaluate
python test_fsl.py   --shot 5 --eval_shot 5 --num_test_episodes 1000   --test_model /home/liuziyi/cub/trainer_ensemble/checkpoints/subsets/MiniImageNet-ProtoNet-Res12-05w05s15q-Pre-DIS/Distill:PN_Coo_10_lr1e-05_T120.0T232_5w_5s-FBN --gpu 0

