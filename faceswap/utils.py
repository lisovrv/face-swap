import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from faceswap.AdaptiveWingLoss.utils.utils import get_preds_fromhm
import torch.nn.functional as F


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def torch2image(torch_image: torch.tensor) -> np.ndarray:
    batch = False
    
    if torch_image.dim() == 4:
        torch_image = torch_image[:8]
        batch = True
    
    device = torch_image.device
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    
    denorm_image = (std * torch_image) + mean
    
    if batch:
        denorm_image = denorm_image.permute(0, 2, 3, 1)
    else:
        denorm_image = denorm_image.permute(1, 2, 0)
    
    np_image = denorm_image.detach().cpu().numpy()
    np_image = np.clip(np_image*255., 0, 255).astype(np.uint8)
    
    if batch:
        return np.concatenate(np_image, axis=1)
    else:
        return np_image


def make_image_list(images) -> np.ndarray:    
    np_images = []
    
    for torch_image in images:
        np_img = torch2image(torch_image)
        np_images.append(np_img)
    
    return np.concatenate(np_images, axis=0)


def read_torch_image(path: str) -> torch.tensor:
    
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = Image.fromarray(image[:, :, ::-1])
    image = transformer_Arcface(image)
    image = image.view(-1, image.shape[0], image.shape[1], image.shape[2])
    
    return image


def get_faceswap(source_path: str, target_path: str, 
                 G: 'generator model', netArc: 'arcface model', 
                 device: 'torch device') -> np.array:
    source = read_torch_image(source_path)
    source = source.to(device)

    embeds = netArc(F.interpolate(source, [112, 112], mode='bilinear', align_corners=False))

    target = read_torch_image(target_path)
    target = target.cuda()

    with torch.no_grad():
        Yt, _ = G(target, embeds)
        Yt = torch2image(Yt)

    source = torch2image(source)
    target = torch2image(target)

    return np.concatenate((cv2.resize(source, (256, 256)), target, Yt), axis=1)  
        


transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def detect_landmarks(inputs, model_ft):
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(inputs.device)
    std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(inputs.device)
    inputs = (std * inputs) + mean

    outputs, boundary_channels = model_ft(inputs)    
    pred_heatmap = outputs[-1][:, :-1, :, :].cpu() 
    pred_landmarks, _ = get_preds_fromhm(pred_heatmap)
    landmarks = pred_landmarks*4.0
    eyes = torch.cat((landmarks[:,96,:], landmarks[:,97,:]), 1)
    return eyes, pred_heatmap[:,96,:,:], pred_heatmap[:,97,:,:]


def paint_eyes(images, eyes):
    list_eyes = []
    for i in range(len(images)):
        mask = torch2image(images[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        
        cv2.circle(mask, (int(eyes[i][0]),int(eyes[i][1])), radius=3, color=(0,255,255), thickness=-1)
        cv2.circle(mask, (int(eyes[i][2]),int(eyes[i][3])), radius=3, color=(0,255,255), thickness=-1)
        
        mask = mask[:, :, ::-1]
        mask = transforms_base(Image.fromarray(mask))
        list_eyes.append(mask)
    tensor_eyes = torch.stack(list_eyes)
    return tensor_eyes