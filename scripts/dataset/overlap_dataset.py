import argparse
import glob
import os
import random

import shutil
from tqdm import tqdm


def get_scenes(paths):
    scenes = []
    for path in paths:
        scenes.append(path.split("/")[-1].split(".")[0])
    return scenes


def check_overlap(path):
    files = glob.glob(os.path.join(path, "*json.gz"))
    scenes = get_scenes(files)

    hm3d_splits = {
        "hm3d-tiny": ["QVAA6zecMHu", "W9YAR9qcuvN", "DoSbsoo4EAg", "4vwGX7U38Ux", "Jfyvj3xn2aJ", "1S7LAXRdDqK", "iKFn6fzyRqs", "YMNvYDhK8mB", "u9rPN5cHWBg", "ixTj1aTMup2", "ggNAcMh8JPT", "wsAYBFtQaL7", "gjhYih4upQ9", "Z2DQddYp1fn", "TYDavTf8oyy", "b3WpMbPFB6q", "226REUyJh2K", "HfMobPm86Xn", "U3oQjwTuMX8", "5biL7VEkByM", "LcAd9dhvVwh", "PPTLa8SkUfo", "j6fHrce9pHR", "gmuS7Wgsbrx", "XiJhRLvpKpX"],
        "hm3d-small": ["1S7LAXRdDqK", "DoSbsoo4EAg", "NEVASPhcrxR", "Wo6kuutE9i7", "gmuS7Wgsbrx", "u9rPN5cHWBg", "1UnKg1rAb8A", "E1NrAhMoqvB", "NGyoyh91xXJ", "XiJhRLvpKpX", "h6nwVLpAKQz", "v7DzfFFEpsD", "226REUyJh2K", "FRQ75PjD278", "NtnvZSMK3en", "YHmAkqgwe2p", "hWDDQnSDMXb", "vDfkYo5VqEQ", "3CBBjsNkhqW", "FnDDfrBZPhh", "PPTLa8SkUfo", "YJDUB7hWg9h", "iKFn6fzyRqs", "vLpv2VX547B", "3XYAD64HpDr", "GGBvSFddQgs", "QN2dRqwd84J", "YMNvYDhK8mB", "iigzG1rtanx", "wPLokgvCnuk", "4vwGX7U38Ux", "GTV2Y73Sn5t", "QVAA6zecMHu", "YmWinf3mhb5", "ixTj1aTMup2", "wsAYBFtQaL7", "5biL7VEkByM", "GtM3JtRvvvR", "RaYrxWt5pR1", "Z2DQddYp1fn", "j6fHrce9pHR", "xAHnY3QzFUN", "6imZUJGRUq4", "HeSYRw7eMtG", "TSJmdttd2GV", "b3WpMbPFB6q", "nACV8wLu1u5", "xWvSkKiWQpC", "77mMEyxhs44", "HfMobPm86Xn", "TYDavTf8oyy", "fxbzYAGkrtm", "nS8T59Aw3sf", "xgLmjqzoAzF", "8wJuSPJ9FXG", "HxmXPBbFCkH", "U3oQjwTuMX8", "g7hUFVNac26", "oEPjPNSPmzL", "yHLr6bvWsVm", "ACZZiU6BXLz", "Jfyvj3xn2aJ", "URjpCob8MGw", "g8Xrdbe9fir", "oahi4u45xMf", "CQWES1bawee", "JptJPosx1Z6", "VoVGtfYrpuQ", "gQ3xxshDiCz", "pcpn6mFqFCg", "CthA7sQNTPK", "LcAd9dhvVwh", "W16Bm4ysK8v", "ggNAcMh8JPT", "qk9eeNeR4vw", "DNWbUAJYsPy", "MVVzj944atG", "W9YAR9qcuvN", "gjhYih4upQ9", "qz3829g1Lzf"],
        "hm3d-medium": ["vLpv2VX547B", "GtM3JtRvvvR", "W16Bm4ysK8v", "VoVGtfYrpuQ", "xAHnY3QzFUN", "iLDo95ZbDJq", "sX9xad6ULKc", "u9rPN5cHWBg", "ooq3SnvC79d", "u5atqC7vRCY", "77mMEyxhs44", "ixTj1aTMup2", "GGBvSFddQgs", "DsEJeNPcZtE", "ggNAcMh8JPT", "wsAYBFtQaL7", "92vYG1q49FY", "NPHxDe6VeCc", "ENiCjXWB6aQ", "gQgtJ9Stk5s", "erXNfWVjqZ8", "HeSYRw7eMtG", "gjhYih4upQ9", "CQWES1bawee", "v7DzfFFEpsD", "Z2DQddYp1fn", "fRZhp6vWGw7", "DqJKU7YU7dA", "JNiWU5TZLtt", "TYDavTf8oyy", "URjpCob8MGw", "KjZrPggnHm8", "fxbzYAGkrtm", "b3WpMbPFB6q", "xgLmjqzoAzF", "8B43pG641ff", "zUG6FL9TYeR", "226REUyJh2K", "HfMobPm86Xn", "nGhNxKrgBPb", "hWDDQnSDMXb", "6imZUJGRUq4", "U3oQjwTuMX8", "DBBESbk4Y3k", "mt9H8KcxRKD", "5biL7VEkByM", "LcAd9dhvVwh", "PPTLa8SkUfo", "JptJPosx1Z6", "UuwwmrTsfBN", "j2EJhFEQGCL", "XVSZJAtHKdi", "L5QEsaVqwrY", "iigzG1rtanx", "bHKTDQFJxTw", "j6fHrce9pHR", "gmuS7Wgsbrx", "XiJhRLvpKpX"],
        "hm3d-large": ["1S7LAXRdDqK", "FRQ75PjD278", "R9fYpvCUkV7", "b3WpMbPFB6q", "nACV8wLu1u5", "1UnKg1rAb8A", "FnDDfrBZPhh", "RTV2n6fXB2w", "bB6nKqfsb1z", "nGhNxKrgBPb", "1xGrZPxG1Hz", "GGBvSFddQgs", "RaYrxWt5pR1", "bHKTDQFJxTw", "nS8T59Aw3sf", "226REUyJh2K", "GPyDUnjwZQy", "S7uMvxjBVZq", "bdp1XNEdvmW", "oEPjPNSPmzL", "2Pc8W48bu21", "GTV2Y73Sn5t", "SgkmkWjjmDJ", "ceJTwFNjqCt", "oPj9qMxrDEa", "3CBBjsNkhqW", "GsQBY83r3hb", "TSJmdttd2GV", "dQrLTxHvLXU", "oStKKWkQ1id", "3XYAD64HpDr", "GtM3JtRvvvR", "TYDavTf8oyy", "erXNfWVjqZ8", "oahi4u45xMf", "4vwGX7U38Ux", "H8rQCnvBgo6", "U3oQjwTuMX8", "fK2vEV32Lag", "ooq3SnvC79d", "5Kw4nGdqYtS", "HZ2iMMBsBQ9", "URjpCob8MGw", "fRZhp6vWGw7", "pcpn6mFqFCg", "5biL7VEkByM", "HeSYRw7eMtG", "UuwwmrTsfBN", "fxbzYAGkrtm", "qZ4B7U6XE5Y", "6HRFAUDqpTb", "HfMobPm86Xn", "VSxVP19Cdyw", "g7hUFVNac26", "qgZhhx1MpTi", "6YtDG3FhNvx", "HkseAnWCgqk", "VoVGtfYrpuQ", "g8Xrdbe9fir", "qk9eeNeR4vw", "6imZUJGRUq4", "HxmXPBbFCkH", "W16Bm4ysK8v", "gQ3xxshDiCz", "qz3829g1Lzf", "741Fdj7NLF9", "JNiWU5TZLtt", "W9YAR9qcuvN", "gQgtJ9Stk5s", "sX9xad6ULKc", "77mMEyxhs44", "Jfyvj3xn2aJ", "WhNyDTnd9g5", "ggNAcMh8JPT", "u5atqC7vRCY", "8B43pG641ff", "JptJPosx1Z6", "Wo6kuutE9i7", "gjhYih4upQ9", "u9rPN5cHWBg", "8wJuSPJ9FXG", "KjZrPggnHm8", "X6Pct1msZv5", "gmuS7Wgsbrx", "v7DzfFFEpsD", "92vYG1q49FY", "L5QEsaVqwrY", "XVSZJAtHKdi", "h6nwVLpAKQz", "vDfkYo5VqEQ", "9h5JJxM6E5S", "LVgQNuK8vtv", "XYyR54sxe6b", "hWDDQnSDMXb", "vLpv2VX547B", "ACZZiU6BXLz", "LcAd9dhvVwh", "XfUxBGTFQQb", "iKFn6fzyRqs", "w8GiikYuFRk", "CQWES1bawee", "MVVzj944atG", "XiJhRLvpKpX", "iLDo95ZbDJq", "wPLokgvCnuk", "CthA7sQNTPK", "NEVASPhcrxR", "YHmAkqgwe2p", "iePHCSf119p", "wsAYBFtQaL7", "DBBESbk4Y3k", "NGyoyh91xXJ", "YJDUB7hWg9h", "iigzG1rtanx", "xAHnY3QzFUN", "DNWbUAJYsPy", "NPHxDe6VeCc", "YMNvYDhK8mB", "ixTj1aTMup2", "xWvSkKiWQpC", "DoSbsoo4EAg", "NtnvZSMK3en", "YY8rqV6L6rf", "j2EJhFEQGCL", "xgLmjqzoAzF", "DqJKU7YU7dA", "PE6kVEtrxtj", "YmWinf3mhb5", "j6fHrce9pHR", "yHLr6bvWsVm", "DsEJeNPcZtE", "PPTLa8SkUfo", "Z2DQddYp1fn", "kA2nG18hCAr", "yX5efd48dLf", "E1NrAhMoqvB", "QN2dRqwd84J", "ZNanfzgCdm3", "kJxT5qssH4H", "zUG6FL9TYeR", "ENiCjXWB6aQ", "QVAA6zecMHu", "aRKASs4e8j1", "mt9H8KcxRKD", "zepmXAdrpjR"]
    }

    hm3d_split_overlap = {}
    for split, split_scenes in hm3d_splits.items():
        overlap = set(split_scenes).intersection(set(scenes))
        print(split, len(overlap))
        hm3d_split_overlap[split] = list(overlap)
    print("\n")
    
    hm3d_split_overlap["hm3d-small"] = hm3d_split_overlap["hm3d-medium"].copy()
    hm3d_split_overlap["hm3d-medium"] = set(hm3d_split_overlap["hm3d-tiny"] + hm3d_split_overlap["hm3d-small"])

    diff = set(hm3d_split_overlap["hm3d-large"]).difference(hm3d_split_overlap["hm3d-medium"])
    sampled = random.sample(list(diff), 10)
    hm3d_split_overlap["hm3d-medium"] = set(list(hm3d_split_overlap["hm3d-medium"]) + sampled)

    for split, split_scenes in hm3d_split_overlap.items():
        print(split, len(split_scenes))
    copy_dataset(hm3d_split_overlap, path)


def copy_dataset(hm3d_splits, path):
    hm3d_split_paths = {
        "hm3d-tiny": "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_tiny/train/content",
        "hm3d-small": "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_small/train/content",
        "hm3d-medium": "data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_medium/train/content",
    }
    for split, split_scenes in hm3d_splits.items():
        if hm3d_split_paths.get(split) is None:
            continue
        output_path = hm3d_split_paths[split]
        for scene in tqdm(split_scenes):
            scene_output_path = os.path.join(output_path, "{}.json.gz".format(scene))
            input_path = os.path.join(path, "{}.json.gz".format(scene))
            try:
                shutil.copy(input_path, scene_output_path)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_data/hits_max_length_1500.json.gz"
    )

    args = parser.parse_args()
    check_overlap(args.path)
