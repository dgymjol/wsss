python main.py --work-dir work_dir/voc/test --config ./config/voc/resnet50_cam_m.yaml

python evaluation.py --weights work_dir/voc/resnet50:15 --device 0 1 --config config/voc/resnet50_cam_m_eval.yaml

lrs=(0.01 0.1 0.001 0.0001)

for lr in ${lrs[@]}
    do
        # python main.py --work-dir work_dir/voc/resnet50_${lr} --config ./config/voc/resnet50_cam.yaml --base-lr ${lr}
        python main.py --work-dir work_dir/voc/new_resnet50_${lr} --config ./config/voc/resnet50_cam_m.yaml --base-lr ${lr}

#         python main.py --work-dir work_dir/cub/r50cam_${lr} --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --num-epoch 95
#         python main.py --work-dir work_dir/cub/r50cam_aug_${lr} --config ./config/cub/resnet50_cam_aug.yaml --base-lr ${lr} --num-epoch 95

    done

