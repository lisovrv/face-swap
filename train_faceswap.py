import torch
import wandb

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from faceswap.trainer import Trainer
from faceswap.dataset import FaceEmbedVGG2
from faceswap.models import Generator, Discriminator, IResNet_100
from faceswap.AdaptiveWingLoss.core import models


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    if config.use_wandb:
        run = wandb.init(config=config, project="faceswap-ghost", name="exp_0")
        run.log_code("./", include_fn=lambda path: path.endswith(".yaml"))


    train_dataset = FaceEmbedVGG2(config.dataset.dataset_path,
                                  same_prob=config.dataset.same_person,
                                  same_identity=config.dataset.same_identity)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=config.num_workers)
    test_dataloader = DataLoader(dataset=train_dataset, batch_size=4,
                                 shuffle=False, drop_last=True,
                                 num_workers=4)

    generator = Generator(config.models.backbone, num_blocks=config.models.num_blocks, c_id=512)
    discr = Discriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d)
    
    if config.models.pretrained:
        try:
            generator.load_state_dict(torch.load(config.models.gen_path,
                                                 map_location=torch.device('cpu')), strict=True)
            discr.load_state_dict(torch.load(config.models.discr_path,
                                             map_location=torch.device('cpu')), strict=True)
            print('Weights loaded')
        except FileNotFoundError as e:
            print('Pretrained weights not found')

    gen_opt = torch.optim.Adam(generator.parameters(), config.optimizers.gen_lr,
                               betas=(0, 0.999), weight_decay=1e-4)
    discr_opt = torch.optim.Adam(discr.parameters(), config.optimizers.discr_lr,
                                 betas=(0, 0.999), weight_decay=1e-4)
    
    if config.scheduler:
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt,
                                                        step_size=config.scheduler_step,
                                                        gamma=config.scheduler_gamma)
        discr_scheduler = torch.optim.lr_scheduler.StepLR(discr_opt,
                                                          step_size=config.scheduler_step,
                                                          gamma=config.scheduler_gamma)
    else:
        gen_scheduler, discr_scheduler = None, None
        
        
    # for id-loss
    arcface_network = IResNet_100(fp16=False)
    arcface_network.load_state_dict(torch.load(os.path.join(orig_cwd,
                                                            'pretrained_models/arcface_backbone.pth')))
    
    # for eye-loss
    if config.eye_detector_loss:
        eye_detector = models.FAN(4, "False", "False", 98)
        checkpoint = torch.load('faceswap/adaptive_wing_loss/AWL_detector/WFLW_4HG.pth')
        if 'state_dict' not in checkpoint:
            eye_detector.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = eye_detector.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            eye_detector.load_state_dict(model_weights)
    else:
        eye_detector=None        
        
    # train
    trainer = Trainer(config, train_dataloader, test_dataloader,
                      generator, discr,
                      gen_opt, discr_opt,
                      gen_scheduler, discr_scheduler,
                      len(train_dataset),
                      arcface_network,
                      eye_detector)
    trainer.train()


if __name__ == "__main__":
    main()
