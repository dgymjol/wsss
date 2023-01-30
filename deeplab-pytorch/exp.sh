python main.py train --config-path configs/voc12.yaml > train.txt

python main.py test --config-path configs/voc12.yaml --model-path data/models/voc12/deeplabv2_resnet101/train_aug_pseudo/checkpoint_final.pth > test.txt

python main.py crf  --config-path configs/voc12.yaml > crf.txt

