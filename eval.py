# Copyright (C) 2019 Klaas Dijkstra
#
# This file is part of OpenCentroidNet.
#
# OpenCentroidNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# OpenCentroidNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with OpenCentroidNet.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
from tqdm import tqdm
import torch.utils.data
import pandas as pd
from config import Config
from argparse import ArgumentParser
from centroidnet import *
from utils.metrics import counting_metrics
from collections import Counter

dev = "cuda:1"


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        help="input validataion image list file",
                        required=True,
                        type=str)
    parser.add_argument("-o",
                        "--output",
                        help="output path",
                        required=False,
                        default='out',
                        type=str)

    return parser


def create_centroidnet(num_channels, num_classes):
    model = centroidnet.CentroidNet(num_classes, num_channels)
    return model


def load_model(filename, model):
    print(f"Load snapshot from: {os.path.abspath(filename)}")
    with open(filename, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    return model


def predict(image, model, max_dist, binning, nm_size, centroid_threshold, sub,
            div):
    # Prepare network input
    inputs = np.expand_dims(np.transpose(image, (2, 0, 1)),
                            axis=0).astype(np.float32)
    inputs = torch.Tensor((inputs - sub) / div)

    # Upload to device
    inputs = inputs.to(Config.dev)
    model.to(Config.dev)

    # Do inference and decoding
    outputs = model(inputs)[0].cpu().detach().numpy()
    centroid_vectors, votes, class_ids, class_probs, votes_nm, centroids = centroidnet.decode(
        outputs, max_dist, binning, nm_size, centroid_threshold)

    # Only return the list of centroids
    return centroids


if __name__ == '__main__':
    args = build_argparser().parse_args()
    os.makedirs(args.output, exist_ok=True)
    if not os.path.exists(args.input):
        print(f"{args.input} is invalid")
        sys.exit(1)
    val_df = pd.read_csv(args.input, header=None)
    file_list = list(set(val_df[0]))
    class_num = len(set(val_df[5]))
    class_labels = list(set(val_df[5]))
    ometrics = counting_metrics(class_num, class_labels, args.output)
    model = create_centroidnet(num_channels=Config.num_channels,
                               num_classes=Config.num_classes)
    model = load_model(os.path.join("data", "CentroidNet.pth"), model)

    for imgi in tqdm(file_list):
        img_path = os.path.join(os.path.dirname(args.input), imgi)
        image = cv2.imread(img_path)
        assert image is not None, f"Image {os.path.abspath(args.input)} not found"

        centroids = predict(image, model, Config.max_dist, Config.binning,
                            Config.nm_size, Config.centroid_threshold,
                            Config.sub, Config.div)
        centroids = np.stack(centroids, axis=0)
        for p in centroids:
            cv2.circle(image, (int(p[1]), int(p[0])), 0, (255, 0, 0))
            cv2.putText(image, f'#{int(p[2])}', (int(p[1]), int(p[0])),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.output, imgi), image)

        pd_result = Counter(centroids[:, 2].astype(np.uint8))
        pd_result['fn'] = imgi
        fdf = val_df[val_df[0] == imgi]
        gt_info = Counter(fdf[5])
        gt_info['fn'] = imgi
        ometrics.add_result(pd_result, gt_info)
    ometrics.calc_metrics()
