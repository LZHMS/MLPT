import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum
from dassl.utils import read_json, mkdir_if_missing

from .oxford_pets import OxfordPets
from .datasetbase import COOPDatasetBase

@DATASET_REGISTRY.register()
class OxfordFlowers(COOPDatasetBase):

    dataset_dir = "oxford_flowers"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.split_expand_dir = os.path.join(self.dataset_dir, "split_expand")
        mkdir_if_missing(self.split_fewshot_dir)
        mkdir_if_missing(self.split_expand_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots, num_fp, num_expand = cfg.DATASET.NUM_SHOTS, cfg.DATASET.NUM_FP, cfg.DATASET.NUM_EXPAND
        if num_shots >= 1:
            seed = cfg.SEED

            fewshot_preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-numfp_{num_fp}-seed_{seed}.pkl")
            expand_preprocessed = os.path.join(self.split_expand_dir, f"shot_{num_expand*num_shots}-numfp_{num_expand*num_fp}-seed_{seed}.pkl")

            # make noise for few-shots dataset 
            if os.path.exists(fewshot_preprocessed):
                print(f"Loading preprocessed noisy few-shot data from {fewshot_preprocessed}")
                with open(fewshot_preprocessed, "rb") as file:
                    data = pickle.load(file)
                    noise_few_train, val = data["train"], data["val"]
            else:
                few_train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                noise_few_train = self.make_noise(few_train, num_fp)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": noise_few_train, "val": val}
                print(f"Saving preprocessed noisy few-shot data to {fewshot_preprocessed}")
                with open(fewshot_preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

            # make noise for train dataset
            if os.path.exists(expand_preprocessed):
                print(f"Loading preprocessed noisy expand data from {expand_preprocessed}")
                with open(expand_preprocessed, "rb") as file:
                    data = pickle.load(file)
                    noise_train = data["train"]
            else:
                few_train_samples =  self.generate_fewshot_dataset(train, num_shots=num_expand * num_shots)
                noise_train = self.make_noise(few_train_samples, num_expand * num_fp)
                data = {"train": noise_few_train}
                print(f"Saving preprocessed noisy expand data to {expand_preprocessed}")
                with open(expand_preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        # using whole training dataset as val dataset
        train_x, val, test = OxfordPets.subsample_classes(noise_few_train, noise_train, test, subsample=subsample)

        super().__init__(train_x=train_x, val=val, test=test)

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test
