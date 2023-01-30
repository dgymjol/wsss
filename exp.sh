python main.py --work-dir work_dir/voc/resnet50 --config config/voc/resnet50_cam_m.yaml --device 0
python pesudo_mask_generator.py --weights work_dir/voc/resnet50:15 --config config/voc/resnet50_cam_m_eval.yaml
python pesudo_mask_evaluation.py > pseudo_miou.txt
