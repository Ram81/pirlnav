import argparse
import boto3
import glob
import os

def upload_file(file_path, bucket, dest_path):
    aws_access_key_id = os.environ.get("S3_AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("S3_AWS_SECRET_ACCESS_KEY")
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    if file_path[-1] == "/":
        file_paths = glob.glob(file_path + "/*.mp4")
        for file_path in file_paths:
            file_name = file_path.split("/")[-1]
            # print(file_path)
            dest_file_path = os.path.join(dest_path, file_name)
            response = client.upload_file(file_path, bucket, dest_file_path, ExtraArgs={'ACL':'public-read'})
    else:
        response = client.upload_file(file_path, bucket, dest_path, ExtraArgs={'ACL':'public-read'})
        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=str, default="data/hit_data/visualisation/unapproved_hits.zip"
    )
    parser.add_argument(
        "--s3-path", type=str, default="data/hit_data/unapproved_hits.zip"
    )
    parser.add_argument(
        "--bucket", type=str, default="habitat-on-web"
    )

    args = parser.parse_args()
    upload_file(args.file, args.bucket, args.s3_path)