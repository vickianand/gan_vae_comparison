""" 
this script has been copied from - 
https://github.com/CW-Huang/IFT6135H19_assignment/blob/master/assignment3/score_fid.py
and modified
"""

import argparse
import os
import numpy as np
import scipy
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier

SVHN_PATH = "data/svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32), interpolation=2),
                classify_svhn.image_transform,
            ]
        ),
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, num_workers=2
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split="test", download=True, transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


# def iterator_mean(iterator):
#     """ calculate mean over a iterator of nu
#     """
#     mu = 0
#     x_count = 0
#     for batch in iterator:
#         batch_size = batch.shape[0]
#         batch_mu = batch.mean(axis=0)
#         mu = (x_count * mu + batch_size * batch_mu) / (x_count + batch_size)
#         x_count += batch_size

#     return mu


def calculate_fid_score(sample_feature_iterator, testset_feature_iterator):

    sample_features = np.stack([sf for sf in sample_feature_iterator], axis=0)
    testset_features = np.stack([tf for tf in testset_feature_iterator], axis=0)

    mu_p = testset_features.mean(axis=0)
    mu_q = sample_features.mean(axis=0)
    print(f"mu_p.shape: {mu_p.shape}, mu_q.shape: {mu_q.shape}")

    cov_p = np.cov(testset_features, rowvar=False)
    cov_q = np.cov(sample_features, rowvar=False)
    print(f"cov_p.shape: {cov_p.shape}, cov_q.shape: {cov_q.shape}")

    first_order_dist = np.linalg.norm(mu_p - mu_q) ** 2
    second_order_dist = np.trace(cov_p + cov_q - 2 * scipy.linalg.sqrtm(cov_p @ cov_q))

    return first_order_dist + second_order_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a directory of images with the FID score."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svhn_classifier.pt",
        help="Path to feature extraction model.",
    )
    parser.add_argument("directory", type=str, help="Path to image directory")
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location="cpu")
    classifier.eval()

    sample_loader = get_sample_loader(args.directory, PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
