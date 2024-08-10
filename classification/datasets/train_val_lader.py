import os
import time
import torch
from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms
from datasets import train_y_mean, train_y_std, train_cb_mean, train_cb_std, \
    train_cr_mean, train_cr_std
from datasets import train_y_mean_upscaled, train_y_std_upscaled, train_cb_mean_upscaled, train_cb_std_upscaled, \
    train_cr_mean_upscaled, train_cr_std_upscaled
from datasets import train_dct_subset_mean, train_dct_subset_std
from datasets import train_upscaled_static_mean, train_upscaled_static_std
from . import train_y_mean_resized, train_y_std_resized, train_cb_mean_resized, train_cb_std_resized, \
    train_cr_mean_resized, train_cr_std_resized
import datasets.cvtransforms as transforms
#-----------original_transform-------------
# transform=transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             # transforms.RandomHorizontalFlip(),
#             # transforms.ToYCrCb(),
#             # transforms.ChromaSubsample(),
#             # transforms.UpsampleDCT(size=224, T=896, debug=False),
#             transforms.Upscale(upscale_factor=2),
#             #TransformUpscaledDCT
#             transforms.TransformUpscaledDCT(),
#             # transforms.ToTensorDCT(),
#             # transforms.SubsetDCT(channels='64'),
#             transforms.Aggregate(),
#             # transforms.NormalizeDCT(
#             #     train_y_mean_resized,  train_y_std_resized,
#             #     train_cb_mean_resized, train_cb_std_resized,
#             #     train_cr_mean_resized, train_cr_std_resized),
#         ])

transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToYCrCb(),
            # transforms.ChromaSubsample(),
            # transforms.UpsampleDCT(size=224, T=896, debug=False),
            transforms.Upscale(upscale_factor=2),
            #TransformUpscaledDCT
            transforms.Transform_torchdct(),
            # transforms.ToTensorDCT(),
            # transforms.SubsetDCT(channels='64'),
            # transforms.Aggregate(),
            # transforms.NormalizeDCT(
            #     train_y_mean_resized,  train_y_std_resized,
            #     train_cb_mean_resized, train_cb_std_resized,
            #     train_cr_mean_resized, train_cr_std_resized),
        ])


def val_loader(args,valdir):
    val_loader = torch.utils.data.DataLoader(
            ImageFolderDCT(valdir, transform),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    return val_loader

def train_loader(args,traindir):
    train_loader = torch.utils.data.DataLoader(
            ImageFolderDCT(traindir, transform),
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    return train_loader
