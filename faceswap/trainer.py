import torch
import os
from tqdm import tqdm
import wandb
import cv2
import numpy as np
from torchvision.utils import make_grid
import torch.nn.functional as F

import sys
sys.path.append('./faceswap/apex')
from apex import amp
from faceswap.utils import detect_landmarks, paint_eyes, make_image_list, get_faceswap
from faceswap.losses import compute_generator_losses, compute_discriminator_loss


class Trainer(object):

    def __init__(self, config, train_dataloader, test_dataloader,
                 generator, discr,
                 gen_opt, discr_opt,
                 gen_scheduler, discr_scheduler,
                 train_dataset_len,
                 arcface_network,
                 eye_detector):

        self.config = config
        self.train_dataset_len = train_dataset_len

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generator = generator.to(self.device)
        self.discr = discr.to(self.device)
        
        self.arcface_network = arcface_network.to(self.device)
        self.arcface_network.eval()
        
        if self.config.eye_detector_loss:
            self.eye_detector = eye_detector.to(self.device)
            self.eye_detector.eval()

        self.gen_opt, self.discr_opt = gen_opt, discr_opt
        self.gen_scheduler, self.discr_scheduler = gen_scheduler, discr_scheduler
        
        self.generator, self.gen_opt = amp.initialize(self.generator, self.gen_opt,
                                                      opt_level=config.optim_level)
        self.discr, self.discr_opt = amp.initialize(self.discr, self.discr_opt,
                                                    opt_level=config.optim_level)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        self.loss_adv_accumulated = 20.

    def train(self):
        self.generator.train()
        self.generator.train()
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)
        
    def train_epoch(self, epoch):
        pbar = tqdm(self.train_dataloader)
        for iteration, (Xs_orig, Xs, Xt, same_person) in enumerate(pbar):

            Xs_orig = Xs_orig.to(self.device)
            Xs = Xs.to(self.device)
            Xt = Xt.to(self.device)
            same_person = same_person.to(self.device)

            # get the identity embeddings of Xs
            with torch.no_grad():
                embed = self.arcface_network(F.interpolate(Xs_orig,
                                                           [112, 112], mode='bilinear',
                                                           align_corners=False))

            diff_person = torch.ones_like(same_person)
            if self.config.diff_eq_same:
                same_person = diff_person

                
            # Generator training 
            
            self.gen_opt.zero_grad()

            Y, Xt_attr = self.generator(Xt, embed)
            Di = self.discr(Y)
            
            ZY = self.arcface_network(F.interpolate(Y, [112, 112],
                                                    mode='bilinear',
                                                    align_corners=False))
            if self.config.eye_detector_loss:
                Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(Xt, self.eye_detector)
                Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, self.eye_detector)
                
                eye_heatmaps = [Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right]
            else:
                eye_heatmaps = None
                
            (lossG, loss_adv_accumulated, L_adv, \
             L_attr, L_id, L_rec, L_l2_eyes) = compute_generator_losses(self.generator,
                                                                        Y, Xt,
                                                                        Xt_attr, Di,
                                                                        embed, ZY,
                                                                        eye_heatmaps,
                                                                        self.loss_adv_accumulated,
                                                                        diff_person,
                                                                        same_person, self.config)
            with amp.scale_loss(lossG, self.gen_opt) as scaled_loss:
                scaled_loss.backward()
            self.gen_opt.step()

            # Discriminator training
            
            self.discr_opt.zero_grad()

            lossD = compute_discriminator_loss(self.discr, Y, Xs, diff_person)

            with amp.scale_loss(lossD, self.discr_opt) as scaled_loss:
                scaled_loss.backward()

            if (not self.config.discr_force) or (self.loss_adv_accumulated < 4.):
                self.discr_opt.step()
                
            if self.config.scheduler:                
                self.gen_scheduler.step()
                self.discr_scheduler.step()

            # Visualization
            
            total_loss = {
                  'gen/loss_id': L_id.item(),
                  'gen/loss_adv': L_adv.item(),
                  'gen/loss_attr': L_attr.item(),
                  'gen/loss_rec': L_rec.item(),
                
                  'gen/loss_gen': lossG.item(),
                  'discr/loss_discr': lossD.item(),
            }
            
            self.generator.eval()
            self.discr.eval()
            
            step_to_log = epoch * self.train_dataset_len + (iteration + 1) * self.config.batch_size
            if (iteration + 1) % self.config.loss_log_step == 0:
                if self.config.use_wandb:
                    wandb.log(total_loss, step=step_to_log)
                    if self.config.eye_detector_loss:
                        wandb.log({"gen/loss_eyes": L_l2_eyes.item()},
                                  commit=False, step=step_to_log)
                else:
                    print(step_to_log, total_loss)

            if (iteration + 1) % self.config.model_save_step == 0:
                gen_path = os.path.join(self.config.model_save_dir,
                                        self.config.exp_name,
                                        f'{step_to_log}_generator.pth')
                gen_path_lattest = os.path.join(self.config.model_save_dir,
                                                self.config.exp_name,
                                                f'lattest_generator.pth')
                torch.save(self.generator.state_dict(), gen_path)
                torch.save(self.generator.state_dict(), gen_path_lattest)
                
                d_path = os.path.join(self.config.model_save_dir,
                                      self.config.exp_name,
                                      f'{step_to_log}_discr.pth')
                d_path_lattest = os.path.join(self.config.model_save_dir,
                                              self.config.exp_name,
                                              f'lattest_generator.pth')
                torch.save(self.discr.state_dict(), d_path)
                torch.save(self.discr.state_dict(), d_path_lattest)
                
                
            if (iteration + 1) % self.config.wandb_img_step == 0:
                with torch.no_grad():
                    images = [Xs, Xt, Y]
                    if self.config.eye_detector_loss:
                        Xt_eyes_img = paint_eyes(Xt, Xt_eyes)
                        Yt_eyes_img = paint_eyes(Y, Y_eyes)
                        images.extend([Xt_eyes_img, Yt_eyes_img])
                    image = make_image_list(images)
                    if self.config.use_wandb:
                        output = wandb.Image(image, caption=f'{step_to_log}_result')
                        wandb.log({"result": output})
                        
                        
                        res1 = get_faceswap('example_images/training/source1.png',
                                    'example_images/training/target1.png',
                                    self.generator, self.arcface_network, self.device)
                        res2 = get_faceswap('example_images/training/source2.png', 
                                            'example_images/training/target2.png',
                                            self.generator, self.arcface_network, self.device)  
                        res3 = get_faceswap('example_images/training/source3.png',
                                            'example_images/training/target3.png',
                                            self.generator, self.arcface_network, self.device)

                        res4 = get_faceswap('example_images/training/source4.png',
                                            'example_images/training/target4.png',
                                            self.generator, self.arcface_network, self.device)
                        res5 = get_faceswap('example_images/training/source5.png',
                                            'example_images/training/target5.png',
                                            self.generator, self.arcface_network, self.device)  
                        res6 = get_faceswap('example_images/training/source6.png',
                                            'example_images/training/target6.png',
                                            self.generator, self.arcface_network, self.device)

                        output1 = np.concatenate((res1, res2, res3), axis=0)
                        output2 = np.concatenate((res4, res5, res6), axis=0)
                        output = np.concatenate((output1, output2), axis=1)
                        wandb.log({"our_images":wandb.Image(output,
                                                        caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
                    else:
                        cv2.imwrite(os.path.join(self.config.model_save_dir,
                                                 self.config.exp_name,
                                                 'generated_image.jpg'), image[:,:,::-1])

                
