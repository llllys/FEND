# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='QCNet')
    parser.add_argument('--root', type=str, default='~/ArgoverseDatasets/') # dataset root
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default='0,1,2,3') # only use 1 gpu for validation results generation
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/QCNet_AV2.ckpt')
    
    args = parser.parse_args()
    # args.ckpt_path = 'checkpoints/QCNet-FEND-small-5ep.ckpt'    # with contrastive loss
    # args.ckpt_path = 'checkpoints/QCNet-small-5ep.ckpt'         # without contrastive loss
    QCNet.add_model_specific_args(parser)
    model = QCNet.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, 
        strict=True, 
        # val_prediction_name='val_prediction_0326'
        )
    
    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='val', # change to <train> to obtain offline difficulty score
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    trainer.validate(model, dataloader)
