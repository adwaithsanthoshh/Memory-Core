"""
Tiny-ImageNet Dataset Loader and Management.
Supports standard Split Tiny-ImageNet for Continual Learning (e.g. 20 tasks, 10 classes each).
Handles train and validation sets, downloading, and synthetic smoke-test generation.
"""

import os
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import glob

from data.transforms import get_transforms


class TinyImageNetDataset(Dataset):
    """
    PyTorch Dataset for Tiny-ImageNet single task or batch.
    """
    def __init__(
        self,
        samples: List[Dict],
        root_dir: str,
        transform=None,
        is_train: bool = True,
        smoke_mode: bool = False
    ):
        self.samples = samples
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_transforms(is_train=is_train)
        self.is_train = is_train
        self.smoke_mode = smoke_mode

    def __len__(self):
        return len(self.samples)

    def _generate_synthetic_image(self, sample: Dict) -> Image.Image:
        """
        Deterministic synthetic image generation for smoke-test mode.
        Produces structured patterns based on class label (0-199).
        """
        label = sample['label']
        
        # Create 64x64 image with structured patterns
        img_np = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Background color dependent on class region
        bg_r = (label * 31) % 256
        bg_g = (label * 67) % 256
        bg_b = (label * 113) % 256
        img_np[:, :] = [bg_r, bg_g, bg_b]
        
        # Foreground pattern strongly tied to class label
        r = (label * 179) % 256
        g = (label * 53) % 256
        b = (label * 97) % 256
        
        # Center box unique to object
        box_size = 10 + (label % 5) * 4
        c = 32
        img_np[c-box_size:c+box_size, c-box_size:c+box_size] = [r, g, b]
        
        # Inner marker for instance
        sub_r = (255 - r) % 256
        sub_g = (255 - g) % 256
        sub_b = (255 - b) % 256
        img_np[c-2:c+2, c-2:c+2] = [sub_r, sub_g, sub_b]

        return Image.fromarray(img_np)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample['path'])

        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                if self.smoke_mode:
                    img = self._generate_synthetic_image(sample)
                else:
                    raise
        else:
            if self.smoke_mode:
                img = self._generate_synthetic_image(sample)
            else:
                raise FileNotFoundError(
                    f"Tiny-ImageNet image not found at: {img_path}. "
                    f"Please download using `python run.py --download-data` "
                    f"or run with `--smoke-test`."
                )

        if self.transform is not None:
            img = self.transform(img)

        return img, sample['label'], sample


class TinyImageNetDataManager:
    """
    Manages Tiny-ImageNet parsing, mapping, and task splitting.
    """
    def __init__(
        self,
        root_dir: str = "data/tiny-imagenet-200",
        num_tasks: int = 20,
        smoke_mode: bool = False
    ):
        if not os.path.exists(root_dir):
            alt_root = os.path.join("memorycore", root_dir)
            if os.path.exists(alt_root):
                root_dir = alt_root
                
        self.root_dir = root_dir
        self.num_tasks = num_tasks
        self.smoke_mode = smoke_mode
        self.total_classes = 200
        
        self.class_to_id = {}
        self.id_to_class = {}
        self._load_class_mappings()

    def _load_class_mappings(self):
        """Loads wnids.txt to map folders to integer labels 0-199."""
        wnids_file = os.path.join(self.root_dir, "wnids.txt")
        if os.path.exists(wnids_file):
            with open(wnids_file, "r") as f:
                lines = sorted([line.strip() for line in f if line.strip()])
            for idx, wnid in enumerate(lines):
                self.class_to_id[wnid] = idx
                self.id_to_class[idx] = wnid
        else:
            # Fallback for smoke mode or before download
            for idx in range(200):
                fake_wnid = f"n{idx:08d}"
                self.class_to_id[fake_wnid] = idx
                self.id_to_class[idx] = fake_wnid

    def load_scenario(self, max_tasks: Optional[int] = None, max_samples_per_task: Optional[int] = None):
        """
        Loads all training and validation samples, split into tasks.
        """
        classes_per_task = self.total_classes // self.num_tasks
        tasks_to_load = max_tasks if max_tasks is not None else self.num_tasks
        
        task_samples = [[] for _ in range(tasks_to_load)]
        test_samples = []

        # Generate fake data paths if in smoke mode and directory doesn't exist
        if self.smoke_mode and not os.path.exists(os.path.join(self.root_dir, "train")):
            for t_idx in range(tasks_to_load):
                start_cls = t_idx * classes_per_task
                end_cls = start_cls + classes_per_task
                for c in range(start_cls, end_cls):
                    num_train = max_samples_per_task if max_samples_per_task else 50
                    for i in range(num_train):
                        task_samples[t_idx].append({
                            "path": f"train/fake_cls/images/img_{c}_{i}.JPEG",
                            "label": c, "task_id": t_idx, "is_train": True, "session": f"task_{t_idx}"
                        })
                    num_test = max_samples_per_task if max_samples_per_task else 10
                    for i in range(num_test):
                        test_samples.append({
                            "path": f"val/images/val_{c}_{i}.JPEG",
                            "label": c, "task_id": -1, "is_train": False, "session": "val"
                        })
            return task_samples, test_samples

        # REAL DATA PARSING
        # 1. Parse Training Data
        train_dir = os.path.join(self.root_dir, "train")
        for wnid, c_id in self.class_to_id.items():
            t_idx = c_id // classes_per_task
            if t_idx >= tasks_to_load:
                continue
            
            img_dir = os.path.join(train_dir, wnid, "images")
            if os.path.exists(img_dir):
                images = glob.glob(os.path.join(img_dir, "*.JPEG"))
                if max_samples_per_task:
                    images = images[:max_samples_per_task]
                for img_path in images:
                    rel_path = os.path.relpath(img_path, self.root_dir).replace('\\', '/')
                    task_samples[t_idx].append({
                        "path": rel_path, "label": c_id, "task_id": t_idx, "is_train": True, "session": f"task_{t_idx}"
                    })

        # 2. Parse Validation Data (used as test set for continual learning)
        val_annotations_file = os.path.join(self.root_dir, "val", "val_annotations.txt")
        if os.path.exists(val_annotations_file):
            with open(val_annotations_file, "r") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        wnid = parts[1]
                        if wnid in self.class_to_id:
                            c_id = self.class_to_id[wnid]
                            t_idx = c_id // classes_per_task
                            if t_idx < tasks_to_load:
                                rel_path = f"val/images/{img_name}"
                                test_samples.append({
                                    "path": rel_path, "label": c_id, "task_id": -1, "is_train": False, "session": "val"
                                })
                                
        if max_samples_per_task is not None:
            # Filter test samples appropriately to limit size in smoke testing
            test_samples = test_samples[:(max_samples_per_task * tasks_to_load)]

        return task_samples, test_samples

    def verify_train_test_separation(self, task_samples: List[List[Dict]], test_samples: List[Dict]):
        """Verifies no overlap between train and test."""
        train_paths = set()
        for samples in task_samples:
            for s in samples:
                train_paths.add(s['path'])

        for s in test_samples:
            assert s['path'] not in train_paths, f"Leakage! {s['path']} in train and test."
        return True


def download_official_tiny_imagenet(target_dir: str = "data/tiny-imagenet-200"):
    """
    Downloads and extracts Tiny-ImageNet.
    """
    base_target = os.path.dirname(target_dir) if "tiny-imagenet-200" in target_dir else target_dir
    os.makedirs(base_target, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(base_target, "tiny-imagenet-200.zip")

    print(f"Downloading Tiny-ImageNet (237 MB) from: {url}")
    
    def _progress(count, block_size, total_size):
        percent = min(100, int(count * block_size * 100 / total_size))
        print(f"\rDownloading: {percent}% [{count * block_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB]", end='', flush=True)

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print("\nDownload complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_target)
        print("Extraction complete.")
    except Exception as e:
        print(f"\nDownload failed: {e}")
