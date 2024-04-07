from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
import numpy as np
import random

class COOPDataManager(DataManager):
    def __init__(self,
                cfg,
                custom_tfm_train=None,
                custom_tfm_test=None,
                dataset_wrapper=None):
        super().__init__(cfg, custom_tfm_train, custom_tfm_test, dataset_wrapper)
        self.cfg = cfg
        self.custom_tfm_train = custom_tfm_train
        self.dataset_wrapper = dataset_wrapper
        self.tfm_train = build_transform(self.cfg, is_train=True)

    def split_data_loader(self, pred, prob):
        # get clean probability
        for idx, item in enumerate(self.dataset.val):
            item._wx = prob[idx]

        pred_labeled_idx = pred.nonzero()[0]
        pred_unlabeled_idx = (1-pred).nonzero()[0]
        
        labeled_dataset = np.take(self.dataset.val, pred_labeled_idx, axis=0)
        unlabeled_dataset = np.take(self.dataset.val, pred_unlabeled_idx, axis=0)

        # randomly select few-shots for training
        num_shots, num_fp = self.cfg.DATASET.NUM_SHOTS, self.cfg.DATASET.NUM_FP
        num_tp = num_shots - num_fp

        tp_samples = self.dataset.generate_fewshot_dataset(labeled_dataset, num_shots=num_tp)
        fp_samples = self.dataset.generate_fewshot_dataset(unlabeled_dataset, num_shots=num_fp)

        labeled_loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=tp_samples,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=self.tfm_train,
            is_train=True,
            dataset_wrapper=self.dataset_wrapper
        )
        
        sampler_type_ = self.cfg.DATALOADER.TRAIN_U.SAMPLER
        batch_size_ = self.cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        n_domain_ = self.cfg.DATALOADER.TRAIN_U.N_DOMAIN
        n_ins_ = self.cfg.DATALOADER.TRAIN_U.N_INS

        if self.cfg.DATALOADER.TRAIN_U.SAME_AS_X:
            sampler_type_ = self.cfg.DATALOADER.TRAIN_X.SAMPLER
            batch_size_ = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
            n_domain_ = self.cfg.DATALOADER.TRAIN_X.N_DOMAIN
            n_ins_ = self.cfg.DATALOADER.TRAIN_X.N_INS

        unlabeled_loader = build_data_loader(
            self.cfg,
            sampler_type=sampler_type_,
            data_source=fp_samples,
            batch_size=batch_size_,
            n_domain=n_domain_,
            n_ins=n_ins_,
            tfm=self.tfm_train,
            is_train=True,
            dataset_wrapper=self.dataset_wrapper
        )
        return labeled_loader, unlabeled_loader

