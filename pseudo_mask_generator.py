import os
import numpy as np
import random
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import argparse
import yaml
import pdb
import inspect
import shutil
import argparse
import yaml
from feeders import imutils
import torch.nn.functional as F
import imageio
from feeders.imutils import *
from PIL import Image

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Computer Vision Task Training Processor')

    parser.add_argument(
        '--results-dir',
        default='./results_dir/temp')

    parser.add_argument(
        '--config',
        default='./config/cub/resnet50_cam_m_eval.yaml')


    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='the interval for printing messages (#iteration)')

    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='the interval for storing models (#iteration)')

    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=100,
        help='the number of classes')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default='Nothing',
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--cam-eval-thres',
        type=float,
        default=0.15
    )
    parser.add_argument(
        '--conf-fg-thres',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--conf-bg-thres',
        type=float,
        default=0.05
    )
    
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')

    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')

    # loss
    parser.add_argument('--loss', default='MultilabelLoss', help='type of optimizer')

    return parser
class Processor():

    def __init__(self, arg):
        self.arg = arg

        if not os.path.isdir(self.arg.results_dir):
            os.makedirs(self.arg.results_dir)

        self.print_log("------------------------")
        self.print_log(str(arg))
        self.print_log("------------------------")
        self.save_arg()
        self.init_seed(self.arg.seed)

        self.load_model()
        self.load_data()
        self.load_loss()

        self.model = self.model.cuda(self.output_device)
        
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids = self.arg.device, output_device=self.output_device)

    def import_class(self, import_str):
        mod_str, _sep, class_str = import_str.rpartition('.')
        __import__(mod_str)
        try:
            return getattr(sys.modules[mod_str], class_str)
        except AttributeError:
            raise ImportError(f'Class {class_str} cannot be found')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.results_dir):
            os.makedirs(self.arg.results_dir)
        with open('{}/config.yaml'.format(self.arg.results_dir), 'w') as f:
            f.write(f"# commend line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def init_seed(self, seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_data(self):
        
        self.data_loader = dict()

        Feeder = self.import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(Feeder), self.arg.results_dir)
        # dataset=Feeder(**self.arg.train_feeder_args)
        # breakpoint()
        self.data_loader['train'] = DataLoader(
                                    dataset=Feeder(**self.arg.train_feeder_args),
                                    batch_size=self.arg.batch_size,
                                    shuffle=False,
                                    num_workers=self.arg.num_worker)
        self.data_loader['test'] =  DataLoader(
                                    dataset=Feeder(**self.arg.test_feeder_args),
                                    batch_size=self.arg.batch_size,
                                    shuffle=False,
                                    num_workers=self.arg.num_worker)

    def load_model(self):
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = self.import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.results_dir)
        self.print_log('model : ', Model)
        self.model = Model(**self.arg.model_args)
        
        if self.arg.weights == 'Nothing':
            self.print_log("No pretrained weights loaded")
            # raise Exception("No pretrained weights loaded")
        else: 
            exp_dir, epoch = self.arg.weights.split(':')
            if not os.path.exists(exp_dir):
                self.print_log(f"Error : the dir doesnt exist {exp_dir}")
                raise Exception(f"the dir doesnt exist {exp_dir}")
            else:
                model_weight_file_name = ''
                for run_file in os.listdir(exp_dir) :
                    if f"runs-{epoch}" in run_file  and '.pt' in run_file:
                        model_weight_file_name = run_file
                if model_weight_file_name == '':
                    self.print_log(f'Error : that epoch{epoch} weight file doesnt exist')
                    raise Exception(f'that epoch{epoch} weight file doesnt exist')

                weights = torch.load(os.path.join(exp_dir,model_weight_file_name))
                self.model.load_state_dict(weights, strict=False)
                self.print_log(f"Successful : transfered weights ({os.path.join(exp_dir,model_weight_file_name)})")
 
    def load_loss(self):
        if self.arg.loss == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss().cuda(self.output_device)
        elif self.arg.loss == 'MultilabelLoss':
            self.loss = nn.MultiLabelSoftMarginLoss().cuda(self.output_device)
        else:
            raise Exception(f"There is no {self.arg.loss}. Add it in load_loss().")  

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)

        with open('{}/result.txt'.format(self.arg.results_dir), 'a') as f:
            print(str, file=f)

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist


    def scores(self, label_trues, label_preds, n_class):
        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        valid = hist.sum(axis=1) > 0  # added
        mean_iu = np.nanmean(iu[valid])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(n_class), iu))

        return {
            "Pixel Accuracy": acc,
            "Mean Accuracy": acc_cls,
            "Frequency Weighted IoU": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def eval_train(self):
        self.model.eval()
        
        for batch_idx, item in enumerate(self.data_loader['train']) :
            with torch.no_grad():
                data = item['imgs'] # (3, 512, 512)
                img = item['img']
                gt_cls = item['label'].cuda(self.output_device)[0] # (num_classes) : multi-label
                size = item['size']

                # forward
                strided_up_size = imutils.get_strided_up_size(size, 16)

                # forward
                outputs = [self.model(img[0].cuda(self.output_device))[1] for img in data]
                outputs = [ o[0] + o[1].flip(-1) for o in outputs]

                highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                            mode='bilinear', align_corners=False) for o in outputs]
                highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
                
                # classification
                valid_cat = torch.nonzero(gt_cls)[:, 0]

                highres_cam = highres_cam[valid_cat]
                highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

                # "keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()
                keys = np.pad(valid_cat.cpu().numpy() + 1, (1, 0), mode='constant')

                fg_conf_cam = np.pad(highres_cam.cpu().numpy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=self.arg.conf_fg_thres)
                fg_conf_cam = np.argmax(fg_conf_cam, axis = 0)
                pred = crf_inference_label(img[0], fg_conf_cam, n_labels=keys.shape[0])
                fg_conf = keys[pred]  
                
                bg_conf_cam = np.pad(highres_cam.cpu().numpy(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=self.arg.conf_bg_thres)
                bg_conf_cam = np.argmax(bg_conf_cam, axis = 0)
                pred = crf_inference_label(img[0], bg_conf_cam, n_labels=keys.shape[0])
                bg_conf = keys[pred]             
                   
                # 2. combine confident fg & bg
                conf = fg_conf.copy()
                conf[fg_conf == 0] = 255
                conf[bg_conf + fg_conf == 0] = 0
                
                # save final pseudo mask
                out = Image.fromarray(conf.astype(np.uint8), mode='P')
                out.putpalette(palette)
                out.save(os.path.join(os.path.join('pseudo/images', item['name'][0] +  '_palette.png')))
                imageio.imwrite(os.path.join('data/VOCdevkit/VOC2012/SegmentationPseudoClassAug', item['name'][0] + '.png'), conf.astype(np.uint8))
            
    def start(self):
        self.eval_train()

if __name__ == '__main__':
    parser = get_parser()

    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()

        for k in default_arg.keys():
            if k not in key:
                print(f'Wrong arg : {k}')
                assert(k in key)

        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    processor = Processor(arg)
    processor.start()