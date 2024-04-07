import os
import pickle
import re

from dassl.data.datasets import DATASET_REGISTRY, Datum
from dassl.utils import mkdir_if_missing
from .datasetbase import COOPDatasetBase
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class UCF101(COOPDatasetBase):

    dataset_dir = "ucf101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.split_expand_dir = os.path.join(self.dataset_dir, "split_expand")
        mkdir_if_missing(self.split_fewshot_dir)
        mkdir_if_missing(self.split_expand_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    label = int(label) - 1  # conver to 0-based index
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
            train, val = OxfordPets.split_trainval(trainval)
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

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(impath=impath, label=label, classname=renamed_action)
                items.append(item)

        return items
