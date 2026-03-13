import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class DomainNetFewShot(DatasetBase):
    """DomainNet dataset adapted for few-shot prompt learning.

    This is separate from Dassl's built-in DomainNet (which is for domain
    adaptation) to support few-shot sampling and base/new class subsampling
    used by MaPLe, CoOp, CoCoOp, etc.

    6 domains: clipart, infograph, painting, quickdraw, real, sketch.
    345 classes across all domains.

    Expected structure under dataset root:
        domainnet/
            clipart/
            infograph/
            painting/
            quickdraw/
            real/
            sketch/
            clipart_train.txt
            clipart_test.txt
            ...
    """

    dataset_dir = "domainnet"

    # All 6 domains
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # Use source_domains if specified, otherwise default to "real"
        if cfg.DATASET.SOURCE_DOMAINS:
            domain = cfg.DATASET.SOURCE_DOMAINS[0]
        else:
            domain = "real"

        self.split_path = os.path.join(
            self.dataset_dir, f"split_DomainNet_{domain}.json"
        )

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(
                self.split_path, self.dataset_dir
            )
        else:
            train_file = os.path.join(self.dataset_dir, f"{domain}_train.txt")
            test_file = os.path.join(self.dataset_dir, f"{domain}_test.txt")

            train, val, test = self._read_split_files(train_file, test_file)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir,
                f"shot_{num_shots}-seed_{seed}-{domain}.pkl",
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(
            train, val, test, subsample=subsample
        )

        super().__init__(train_x=train, val=val, test=test)

    def _read_split_files(self, train_file, test_file, p_val=0.2):
        """Read official DomainNet train/test split files.

        Each line in the split file has format:
            domain/classname/image_filename.jpg label_index
        """

        def _parse_file(filepath):
            items = []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ")
                    impath = parts[0]
                    label = int(parts[1])
                    # Extract classname from path: domain/classname/image.jpg
                    classname = impath.split("/")[1]
                    classname = classname.replace("_", " ")
                    full_impath = os.path.join(self.dataset_dir, impath)
                    items.append(Datum(impath=full_impath, label=label, classname=classname))
            return items

        all_train = _parse_file(train_file)
        test = _parse_file(test_file)

        # Split train into train and val
        random.seed(42)
        # Group by class
        class_to_items = {}
        for item in all_train:
            if item.label not in class_to_items:
                class_to_items[item.label] = []
            class_to_items[item.label].append(item)

        train, val = [], []
        for label in sorted(class_to_items.keys()):
            items = class_to_items[label]
            random.shuffle(items)
            n_val = max(1, round(len(items) * p_val))
            val.extend(items[:n_val])
            train.extend(items[n_val:])

        return train, val, test
