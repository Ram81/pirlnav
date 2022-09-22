import argparse
import sys

import subprocess

S3_BUCKET = "habitat-on-web"
S3_CHECKPOINT_PATH = "checkpoints"


def get_s3_path(path):
    return "s3://{}/{}/{}".format(S3_BUCKET, S3_CHECKPOINT_PATH, path)


def get_command(local_path, s3_path):
    return "aws s3 sync {} {}".format(local_path, s3_path)


def upload_file(path):
    s3_path = get_s3_path(path)
    command = get_command(path, s3_path)
    print("Executing command: {}".format(command))

    proc = subprocess.Popen(command, shell=True, stdout=sys.stdout)
    proc.wait()

    print("Checkpoints uploaded: {}!".format(proc.returncode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_data/visualisation/unapproved_hits.zip"
    )

    args = parser.parse_args()
    upload_file(args.path)
