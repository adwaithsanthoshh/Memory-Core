"""
CORe50 Dataset Loader and Management.
Supports official CORe50 benchmark scenarios: NIC_v2_79, NI, NC.
Handles official test set (sessions s3, s7, s10) and training sessions.
Provides deterministic smoke-test generator for rapid offline verification.
"""

import os
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple

from data.transforms import get_transforms


class CORe50Dataset(Dataset):
    """
    PyTorch Dataset for CORe50 single task or batch.
    """
    def __init__(
        self,
        samples: List[Dict],
        root_dir: str,
        transform = None,
        is_train: bool = True,
        smoke_mode: bool = False
    ):
        """
        Args:
            samples: List of dicts with 'path', 'label', 'session', 'object_id', 'class_name', 'category_name'.
            root_dir: Path to directory containing core50 images.
            transform: PyTorch transforms to apply.
            is_train: Whether this is training set.
            smoke_mode: If True and image file not found, synthetically render deterministic image.
        """
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
        Produces structured patterns based on class label and session
        so a neural network can genuinely learn and demonstrate forgetting.
        """
        label = sample['label']
        # Session digit
        sess_num = int(sample['session'].replace('s', '')) if 's' in sample['session'] else 1
        
        # Create 128x128 image with structured patterns
        img_np = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Background color dependent on session (domain shift)
        bg_r = (sess_num * 23) % 256
        bg_g = (sess_num * 47) % 256
        bg_b = (sess_num * 71) % 256
        img_np[:, :] = [bg_r, bg_g, bg_b]
        
        # Foreground pattern strongly tied to class label
        r = (label * 53) % 256
        g = (label * 97) % 256
        b = (label * 179) % 256
        
        # Center box unique to object
        box_size = 20 + (label % 5) * 8
        c = 64
        img_np[c-box_size:c+box_size, c-box_size:c+box_size] = [r, g, b]
        
        # Inner marker for instance
        sub_r = (255 - r)
        sub_g = (255 - g)
        sub_b = (255 - b)
        img_np[c-4:c+4, c-4:c+4] = [sub_r, sub_g, sub_b]

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
                    f"CORe50 image not found at: {img_path}. "
                    f"Please download CORe50 using `python run.py --download-data` "
                    f"or run with `--smoke-test` for quick offline verification."
                )

        if self.transform is not None:
            img = self.transform(img)

        return img, sample['label'], sample


class CORe50DataManager:
    """
    Manages official CORe50 benchmark metadata, task splitting, and downloads.
    """
    OFFICIAL_TEST_SESSIONS = {'s3', 's7', 's10'}
    OFFICIAL_TRAIN_SESSIONS = {'s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11'}

    def __init__(
        self,
        root_dir: str = "data/core50_128x128",
        metadata_dir: str = "data/core50_metadata",
        scenario: str = "NIC_v2_79",
        run_id: int = 0,
        smoke_mode: bool = False
    ):
        # Resolve metadata path whether running from repo root or memorycore package
        if not os.path.exists(metadata_dir):
            pkg_meta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core50_metadata")
            alt_meta = os.path.join("memorycore", metadata_dir)
            if os.path.exists(pkg_meta):
                metadata_dir = pkg_meta
            elif os.path.exists(alt_meta):
                metadata_dir = alt_meta

        if not os.path.exists(root_dir):
            alt_root = os.path.join("memorycore", root_dir)
            if os.path.exists(alt_root):
                root_dir = alt_root

        self.root_dir = root_dir
        self.metadata_dir = metadata_dir
        self.scenario = scenario
        self.run_id = run_id
        self.smoke_mode = smoke_mode
        
        self.class_names = self._load_class_names()
        self.category_names = self._build_category_map()

    def _load_class_names(self) -> List[str]:
        """Loads 50 class names from core50_labels.txt."""
        labels_file = os.path.join(self.metadata_dir, "core50_labels.txt")
        if os.path.exists(labels_file):
            with open(labels_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) == 50:
                return lines
        # Default official CORe50 50 object names fallback
        categories = ["plug_adapter", "mobile_phone", "scissor", "light_bulb", 
                      "can", "glass", "ball", "marker", "cup", "remote_control"]
        names = []
        for cat in categories:
            for i in range(1, 6):
                names.append(f"{cat}{i}")
        return names

    def _build_category_map(self) -> Dict[int, str]:
        """Maps class_id (0..49) to semantic category name."""
        cat_map = {}
        for idx, name in enumerate(self.class_names):
            # strip trailing digits
            cat_name = ''.join([c for c in name if not c.isdigit()])
            cat_map[idx] = cat_name
        return cat_map

    def _parse_filelist(self, filepath: str) -> List[Dict]:
        """
        Parses official CORe50 filelist.
        Format per line: s1/o1/C_01_01_000.png 0
        """
        samples = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                label = int(parts[1])

                # Parse session and object_id from path
                # Path structure: session/object/filename (e.g., s1/o1/C_01_01_000.png)
                path_parts = rel_path.replace("\\", "/").split("/")
                session = path_parts[0] if len(path_parts) > 1 else "unknown"
                object_id = path_parts[1] if len(path_parts) > 2 else "unknown"

                samples.append({
                    "path": rel_path,
                    "label": label,
                    "session": session,
                    "object_id": object_id,
                    "class_name": self.class_names[label] if label < len(self.class_names) else f"class_{label}",
                    "category_name": self.category_names.get(label, "unknown")
                })
        return samples

    def get_task_filelists(self) -> Tuple[List[str], str]:
        """
        Returns list of train task filelist paths and test filelist path
        for the specified scenario and run_id.
        """
        run_folder = f"run{self.run_id}"
        scenario_dir = os.path.join(self.metadata_dir, self.scenario, run_folder)
        
        if not os.path.exists(scenario_dir):
            raise FileNotFoundError(
                f"Benchmark metadata folder missing: {scenario_dir}. "
                f"Ensure batches_filelists_NICv2.zip is downloaded."
            )

        train_files = []
        num_batches = 79 if "79" in self.scenario else 79
        for i in range(num_batches):
            filename = f"train_batch_{i:02d}_filelist.txt"
            path = os.path.join(scenario_dir, filename)
            if os.path.exists(path):
                train_files.append(path)
            else:
                raise FileNotFoundError(f"Missing batch filelist: {path}")

        test_path = os.path.join(scenario_dir, "test_filelist.txt")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing test filelist: {test_path}")

        return train_files, test_path

    def load_scenario(self, max_tasks: Optional[int] = None, max_samples_per_task: Optional[int] = None):
        """
        Loads all tasks and the official test set.
        
        Args:
            max_tasks: Optional limit on number of tasks (for quick debugging/smoke testing).
            max_samples_per_task: Optional limit on samples per task (for smoke testing).
        """
        train_filelists, test_filelist = self.get_task_filelists()
        
        if max_tasks is not None:
            train_filelists = train_filelists[:max_tasks]

        task_samples = []
        for task_idx, fpath in enumerate(train_filelists):
            samples = self._parse_filelist(fpath)
            for s in samples:
                s['task_id'] = task_idx
                s['is_train'] = True
            if max_samples_per_task is not None and len(samples) > max_samples_per_task:
                samples = samples[:max_samples_per_task]
            task_samples.append(samples)

        test_samples = self._parse_filelist(test_filelist)
        for s in test_samples:
            s['task_id'] = -1
            s['is_train'] = False

        if max_samples_per_task is not None and len(test_samples) > (max_samples_per_task * len(train_filelists)):
            test_samples = test_samples[:(max_samples_per_task * len(train_filelists))]

        return task_samples, test_samples

    def verify_train_test_separation(self, task_samples: List[List[Dict]], test_samples: List[Dict]):
        """
        Strictly verifies that:
        1. No test paths appear in training sets.
        2. No test sessions (s3, s7, s10) appear in training samples.
        3. All test samples belong strictly to test sessions.
        """
        train_paths = set()
        train_sessions = set()
        for t_idx, samples in enumerate(task_samples):
            for s in samples:
                train_paths.add(s['path'])
                train_sessions.add(s['session'])
                assert s['session'] not in self.OFFICIAL_TEST_SESSIONS, (
                    f"Data leakage detected! Training task {t_idx} contains test session {s['session']}: {s['path']}"
                )

        test_paths = set()
        for s in test_samples:
            test_paths.add(s['path'])
            assert s['session'] in self.OFFICIAL_TEST_SESSIONS, (
                f"Test set contains non-test session {s['session']}: {s['path']}"
            )
            assert s['path'] not in train_paths, (
                f"Data leakage detected! Test sample {s['path']} found in training data!"
            )

        overlap = train_paths.intersection(test_paths)
        assert len(overlap) == 0, f"Critical violation: {len(overlap)} samples overlap between train and test!"
        return True


def download_official_core50(target_dir: str = "data/core50_128x128"):
    """
    Downloads and extracts official core50_128x128.zip.
    """
    os.makedirs(target_dir, exist_ok=True)
    url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
    zip_path = os.path.join(target_dir, "core50_128x128.zip")

    print(f"Downloading official CORe50 (128x128) from: {url}")
    print(f"Target path: {zip_path}")
    print("This is ~1.2 GB and may take several minutes depending on internet connection.")

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}% [{count * block_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB]", end='')

    urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    print("\nDownload complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")
