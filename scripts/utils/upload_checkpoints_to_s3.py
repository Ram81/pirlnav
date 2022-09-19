import argparse
import boto3
import glob
import os

from tqdm import tqdm

S3_BUCKET = "habitat-on-web"
S3_CHECKPOINT_PATH = "checkpoints"


def get_s3_client():
    aws_access_key_id = os.environ.get("S3_AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("S3_AWS_SECRET_ACCESS_KEY")
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    return client


def upload_file(path):
    client = get_s3_client()

    checkpoints = glob.glob(os.path.join(path, "*pth"))

    count = 0
    for ckpt_file in tqdm(checkpoints):
        checkpoint_s3_path = os.path.join(S3_CHECKPOINT_PATH, ckpt_file)
        response = client.upload_file(ckpt_file, S3_BUCKET, checkpoint_s3_path, ExtraArgs={'ACL':'public-read'})
        count += 1

    print("Successfully uploaded {} checkpoints!".format(count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_data/visualisation/unapproved_hits.zip"
    )

    args = parser.parse_args()
    upload_file(args.path)
