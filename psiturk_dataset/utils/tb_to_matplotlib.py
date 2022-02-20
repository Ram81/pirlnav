import argparse
from os import write
import numpy as np

import glob
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from psiturk_dataset.utils.utils import write_json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage.filters import gaussian_filter1d


def get_metrics(path):
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    metrics = []
    files = glob.glob(path)
    metric_step_map = {}
    print("Total logs: {}".format(len(files)))
    for path in files:
        epoch = int(path.split("/")[-3].split("_")[-1])

        event_acc = EventAccumulator(path, tf_size_guidance)
        event_acc.Reload()

        metric =  event_acc.Scalars('eval_metrics')
        metrics.append(metric[0][2])
        metric_step_map[epoch] = metric[0][2]

    return metric_step_map


def fetch_metrics(path):
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }
    files = glob.glob(path)
    event_acc = EventAccumulator(files[0], tf_size_guidance)
    event_acc.Reload()

    metric =  event_acc.Scalars('eval_metrics')

    return metric[0][2]


def plot_tensorflow_log(path, output_path):

    metrics = ["eval_metrics_success", "eval_metrics_spl", "eval_metrics_cross_entropy"]
    metric_dict = {}
    for metric in metrics:
        metric_path = path + "/{}/*".format(metric)
        extracted_metrics = get_metrics(metric_path)
        metric_name = metric.split("_")[-1]
        for k, v in extracted_metrics.items():
            if k not in metric_dict.keys():
                metric_dict[k] = {}
            metric_dict[k][metric_name] = v

    print(metric_dict)
    df = pd.DataFrame.from_dict(metric_dict, orient="index")
    df["ckpt"] = df.index
    df = df.sort_values(by="ckpt")
    stats = df.to_dict("list")
    write_json(stats, output_path)
    print(df.to_dict("list"))


def collect_tensorflo_logs(output_path):
    metrics = ["eval_metrics_success", "eval_metrics_spl"]
    metric_dict = {}

    paths = {
        "9600": ["tb/resnet18_random_split_v4/seed2_1node/val", "tb/resnet18_random_split_v4/seed2_1node/test"],
        "5000": ["tb/resnet18_random_split_v5/seed2_1node/val", "tb/resnet18_random_split_v5/seed2_1node/test"],
        "2500": ["tb/resnet18_random_split_v6/seed2_1node/val", "tb/resnet18_random_split_v6/seed2_1node/test"]
    }
    for d_split, paths in paths.items():
        metric_dict[int(d_split)] = {}
        for path in paths:
            split = path.split("/")[-1]
            for metric in metrics:
                metric_path = path + "/{}/*".format(metric)
                extracted_metric = fetch_metrics(metric_path)
                metric_name = "{}_{}".format(split, metric.split("_")[-1])

                metric_dict[int(d_split)][metric_name] = extracted_metric

    df = pd.DataFrame.from_dict(metric_dict, orient="index")
    df["split"] = df.index
    df = df.sort_values(by="split")
    stats = df.to_dict("list")
    write_json(stats, output_path)
    print(df.to_dict("list"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/stats/pick_and_place_dataset_size_perf.json"
    )
    parser.add_argument(
        "--collect", dest='collect', action='store_true'
    )
    args = parser.parse_args()
    if args.collect:
        collect_tensorflo_logs(args.output_path)
    else:
        plot_tensorflow_log(args.log_path, args.output_path)