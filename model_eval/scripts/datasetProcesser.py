import os
import json
import shutil
from typing import Dict, List, Tuple, Optional

# Meta file of all datasets (label_set, train/test_meta, tasktype)
DATASET_META_CLS = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json"
# Base directory of train/test_meta
META_DATA_BASE = "/jhcnas4/Generalist/shebd/Generalist_Meta_data"
# Base directory of datasets
DATASET_BASE = "/jhcnas4/Generalist/Medical_Data_2025"
# Local directory of datasets
LOCAL_DATASET_BASE = "/home/sunanhe/luoyi/model_eval/datasets"
# Selected datasets to process
SELECTED_DATASETS = ["037_WCE", "038_HAM10000", "039_RFMiD", "043_UBIBC"]

class DatasetProcesser:
    """
    Creates local train/test folder structure
    """
    
    def __init__(self,
                 selected_datasets: List[str],
                 dataset_meta_cls: str = DATASET_META_CLS,
                 meta_data_base_path: str = META_DATA_BASE,
                 dataset_base_path: str = DATASET_BASE,
                 local_dataset_base_path: str = LOCAL_DATASET_BASE):
        
        self.selected_datasets = selected_datasets
        self.dataset_meta_path = dataset_meta_cls
        self.meta_data_base_path = meta_data_base_path
        self.dataset_base_path = dataset_base_path
        self.local_dataset_base_path = local_dataset_base_path
        self.dataset_meta = self._load_json_file(self.dataset_meta_path)
    
    def _load_json_file(self, path) -> Dict:
        """Load dataset metadata from JSON file."""
        with open(path, 'r') as f:
            json_dict = json.load(f)
        return json_dict

    def _load_jsonl_file(self, path) -> List[Dict]:
        """Load train/test set metadata from JSONL file."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _get_local_dataset_size(self, dataset_name: str) -> Tuple[int, int]:
        """Get the number of training and testing samples for a dataset."""
        local_train_path = os.path.join(self.local_dataset_base_path, dataset_name, 'train')
        local_test_path = os.path.join(self.local_dataset_base_path, dataset_name, 'test')
        
        # Count actual image files, not directories
        local_train_num = 0
        local_test_num = 0
        
        if os.path.exists(local_train_path):
            for root, dirs, files in os.walk(local_train_path):
                local_train_num += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        
        if os.path.exists(local_test_path):
            for root, dirs, files in os.walk(local_test_path):
                local_test_num += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
            
        return local_train_num, local_test_num

    def check_dataset_exists_locally(self, dataset_name: str) -> bool:
        """Check if the dataset exists in the local directory."""
        local_train_num, local_test_num = self._get_local_dataset_size(dataset_name)
        meta_train_num, meta_test_num = self.dataset_meta['D'+dataset_name]['train_num'], self.dataset_meta['D'+dataset_name]['test_num']
        return local_train_num == meta_train_num and local_test_num == meta_test_num
    
    def copy_dataset_to_local(self, dataset_name: str) -> None:
        """Copy dataset from the base directory to the local directory."""
        if self.check_dataset_exists_locally(dataset_name):
            return      # If dataset already exists locally, skip copying

        src_path = os.path.join(self.dataset_base_path, dataset_name)
        dst_path = os.path.join(self.local_dataset_base_path, dataset_name)
        # Delete existing local dataset directory if it exists
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)

        # Label set
        label_set = self.dataset_meta['D'+dataset_name]['label_set']
        
        # train_meta, test_meta: List[Dict]
        train_meta = self._load_jsonl_file(os.path.join(self.meta_data_base_path, self.dataset_meta['D'+dataset_name]['train_meta']))
        test_meta = self._load_jsonl_file(os.path.join(self.meta_data_base_path, self.dataset_meta['D'+dataset_name]['test_meta']))
        
        # Create local dataset directory
        os.makedirs(dst_path, exist_ok=True)
        # Create train and test directories
        train_path = os.path.join(dst_path, 'train')
        test_path = os.path.join(dst_path, 'test')
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Copy and reorganize train and test sets to local directories
        if self.dataset_meta['D'+dataset_name]['tasktype'] == 'multilabel':
            for line in train_meta:
                src_image_path = os.path.join(self.dataset_base_path, line['image'])
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'train')
                shutil.copy(src_image_path, dst_image_directory)
            
            for line in test_meta:
                src_image_path = os.path.join(self.dataset_base_path, line['image'])
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'test')
                shutil.copy(src_image_path, dst_image_directory)

        else:
            for line in train_meta:
                image = line['image']

                label_idx = line['label_idx'] # may be [int], need convert to int
                label = label_set[label_idx] if isinstance(label_idx, int) else label_set[label_idx[0]]
                
                src_image_path = os.path.join(self.dataset_base_path, image)
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'train', label)
                os.makedirs(dst_image_directory, exist_ok=True)
                shutil.copy(src_image_path, dst_image_directory)

            for line in test_meta:
                image = line['image']

                label_idx = line['label_idx'] # may be [int], need convert to int
                label = label_set[label_idx] if isinstance(label_idx, int) else label_set[label_idx[0]]

                src_image_path = os.path.join(self.dataset_base_path, image)
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'test', label)
                os.makedirs(dst_image_directory, exist_ok=True)
                shutil.copy(src_image_path, dst_image_directory)
        
    def process_all_selected_datasets(self) -> None:
        """Process all selected datasets."""
        for dataset_name in self.selected_datasets:    
            self.copy_dataset_to_local(dataset_name)



if __name__ == "__main__":
    # Debugging purpose
    processer = DatasetProcesser(SELECTED_DATASETS)
    processer.process_all_selected_datasets()
    for dataset_name in processer.selected_datasets:
        local_train_num, local_test_num = processer._get_local_dataset_size(dataset_name)
        print(f"Dataset: {dataset_name}, Local Train Num: {local_train_num}, Local Test Num: {local_test_num}")
        exists = processer.check_dataset_exists_locally(dataset_name)
        print(f"Dataset {dataset_name} exists locally: {exists}")