import os
import torch
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


class JointTransform:
    def __init__(self, size, normalize=True):
        self.size = size
        self.normalize = normalize

    def __call__(self, img, mask):
        # Convert to tensors
        img = torch.as_tensor(np.array(img), dtype=torch.uint8)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        if img.ndim == 3:
            img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]

        img = F.resize(img, self.size)        
        mask = F.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        
        mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
        
        if self.normalize:
            img = img.float() / 255.0  # Normalize to [0, 1] first
            img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img, mask


class COCOSegmentationDataset(Dataset):
    """
    COCO Format Segmentation Dataset Loader
    
    Args:
        root_dir: Path to the dataset root directory
        split: 'train', 'valid', or 'test' 
        transform: Optional transform to be applied on images and masks
        return_bbox: Whether to return bounding box annotations
        return_category: Whether to return category information
    """
    
    def __init__(self, root_dir, split='train', transform=None, 
                 return_bbox=True, return_category=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_bbox = return_bbox
        self.return_category = return_category
        
        # Load COCO annotations
        ann_file = self.root_dir / split / '_annotations.coco.json'
        self.coco = COCO(ann_file)
        
        # Get all image IDs
        self.image_ids = list(self.coco.imgs.keys())
        
        # Get category information
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.num_classes = len(self.categories)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image info
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = self.root_dir / self.split / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create segmentation mask
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        bboxes = []
        categories = []
        
        for ann in anns:
            # Get category
            category_id = ann['category_id']
            categories.append(category_id)
            
            # Get bounding box
            if self.return_bbox and 'bbox' in ann:
                bbox = ann['bbox']  # [x, y, width, height] in COCO format
                x, y, bbox_w, bbox_h = bbox
                bboxes.append([x, y, x + bbox_w, y + bbox_h])  # Convert to [xmin, ymin, xmax, ymax]
            
            # Create mask from segmentation
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    rles = coco_mask.frPyObjects(ann['segmentation'], h, w)
                    rle = coco_mask.merge(rles)
                elif isinstance(ann['segmentation'], dict):
                    # RLE format
                    rle = ann['segmentation']
                else:
                    continue
                
                m = coco_mask.decode(rle)
                # Use category_id as mask value (for multi-class segmentation)
                mask[m > 0] = category_id
        
        # Convert to PIL Image for transform compatibility
        mask_pil = Image.fromarray(mask)
        
        result = {
            'image': None,
            'mask': None,
            'bbox': bboxes if self.return_bbox else None,
            'category': categories if self.return_category else None,
            'image_id': img_id,
            'file_name': img_info['file_name']
        }
        
        if self.transform:
            image, mask = self.transform(image, mask_pil)
        else:
            image = torch.as_tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1)
            mask = torch.as_tensor(np.array(mask_pil), dtype=torch.long)
        
        result['image'] = image
        result['mask'] = mask

        return result
    
    def get_category_name(self, category_id):
        """Get category name from category ID"""
        return self.categories.get(category_id, 'unknown')


def collate_fn(batch):
    images = []
    masks = []
    bboxes = []
    categories = []
    
    for item in batch:
        images.append(item['image'])
        masks.append(item['mask'])
        bboxes.append(item.get('bbox'))
        categories.append(item.get('category'))
    
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return images, masks, categories, bboxes


def get_coco_loader(root_dir, split='train', batch_size=32, 
                    image_size=224, num_workers=4, shuffle=None, 
                    normalize=True, return_bbox=True, return_category=True,
                    seed=42):
    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Handle image size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    transform = JointTransform(size=image_size, normalize=normalize)
    
    dataset = COCOSegmentationDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        return_bbox=return_bbox,
        return_category=return_category
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
    
    return loader, dataset
