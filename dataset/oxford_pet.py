import os
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F


class JointTransform:
    def __init__(self, size, normalize=True):
        self.size = size
        self.normalize = normalize

    def __call__(self, img, mask, trimap):
        # Convert to tensors
        img = torch.as_tensor(np.array(img), dtype=torch.uint8)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        trimap = torch.as_tensor(np.array(trimap), dtype=torch.long)
        
        if img.ndim == 3:
            img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        if trimap.ndim == 2:
            trimap = trimap.unsqueeze(0)  # [H, W] -> [1, H, W]

        img = F.resize(img, self.size)        
        mask = F.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        trimap = F.resize(trimap, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        
        mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
        trimap = trimap.squeeze(0)  # [1, H, W] -> [H, W]
        
        if self.normalize:
            img = img.float() / 255.0  # Normalize to [0, 1] first
            img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img, mask, trimap

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split='trainval', image_dir=None, 
                 transform=None, load_trimap=True, load_bbox=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.load_trimap = load_trimap
        self.load_bbox = load_bbox
        
        if image_dir is None:
            self.image_dir = self.root_dir.parent / 'images'
        else:
            self.image_dir = Path(image_dir)
            
        self.trimap_dir = self.root_dir / 'trimaps'
        self.xml_dir = self.root_dir / 'xmls'
        
        split_file = self.root_dir / f'{split}.txt'
        self.samples = []
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                image_name = parts[0]
                class_id = int(parts[1])
                species = int(parts[2])  # 1: Cat, 2: Dog
                breed_id = int(parts[3])
                
                self.samples.append({
                    'image_name': image_name,
                    'class_id': class_id,
                    'species': species,
                    'breed_id': breed_id
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = sample['image_name']
        image_path = self.image_dir / f'{image_name}.jpg'
        image = Image.open(image_path).convert('RGB')

        result = {
            'image': None,
            'trimap': None,
            'mask': None,
            'bbox': None,
        }
        
        if self.load_bbox:
            xml_path = self.xml_dir / f'{image_name}.xml'
            if xml_path.exists():
                bbox = self._parse_xml(xml_path)
                result['bbox'] = bbox

        if self.load_trimap:
            trimap_path = self.trimap_dir / f'{image_name}.png'
            if trimap_path.exists():
                trimap = Image.open(trimap_path)
                trimap_array = np.array(trimap)
                mask = (trimap_array == 1).astype(np.uint8)
        
        if self.transform:
            image, mask, trimap = self.transform(image, mask, trimap)
        
        result['image'] = image
        result['trimap'] = trimap
        result['mask'] = mask

        return result
    
    def _parse_xml(self, xml_path):
        """Parse XML file to extract bounding box"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        bbox_elem = root.find('.//bndbox')
        if bbox_elem is not None:
            xmin = int(bbox_elem.find('xmin').text)
            ymin = int(bbox_elem.find('ymin').text)
            xmax = int(bbox_elem.find('xmax').text)
            ymax = int(bbox_elem.find('ymax').text)
            return [xmin, ymin, xmax, ymax]
        return None


def collate_fn(batch):
    images = []
    trimaps = []
    masks = []
    bboxes = []
    
    for item in batch:
        images.append(item['image'])
        trimap = item['trimap']
        mask = item['mask']
        bbox = item.get('bbox')
        
        trimaps.append(trimap)
        masks.append(mask)
        bboxes.append(bbox)
    
    images = torch.stack(images, dim=0)  # [B, C, H, W]
    trimaps = torch.stack(trimaps, dim=0)  # [B, H, W]
    masks = torch.stack(masks, dim=0)  # [B, H, W]
    
    return images, masks, trimaps, bboxes


def get_oxford_pet_loader(root_dir, split='trainval', batch_size=32, 
                          image_size=224, num_workers=4, shuffle=None, seed=42):
    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'trainval')
    
    transform = JointTransform(size=(image_size, image_size), normalize=True)
    
    dataset = OxfordPetDataset(
        root_dir=root_dir,
        split=split,
        transform=transform
    )
    
    # Create DataLoader with reproducible shuffling
    generator = None
    if seed is not None and shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        collate_fn=collate_fn
    )
    
    return loader
