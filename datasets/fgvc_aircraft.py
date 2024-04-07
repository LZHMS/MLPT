import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum
from dassl.utils import mkdir_if_missing
from .oxford_pets import OxfordPets
from .datasetbase import COOPDatasetBase

@DATASET_REGISTRY.register()
class FGVCAircraft(COOPDatasetBase):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.split_expand_dir = os.path.join(self.dataset_dir, "split_expand")
        mkdir_if_missing(self.split_fewshot_dir)
        mkdir_if_missing(self.split_expand_dir)

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

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

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
