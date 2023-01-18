python main.py --work-dir work_dir/voc/resnet50 --config ./config/voc/resnet50_cam.yaml

# python evaluation.py --weights work_dir/cub/r50cam_0.005:67 --device 0 1 --config config/cub/resnet50_cam_eval.yaml

# lrs=(0.001 0.005 0.0025 0.001)

# for lr in ${lrs[@]}
#     do
#         python main.py --work-dir work_dir/cub/r50cam_${lr} --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --num-epoch 95
#         python main.py --work-dir work_dir/cub/r50cam_aug_${lr} --config ./config/cub/resnet50_cam_aug.yaml --base-lr ${lr} --num-epoch 95

#     done

