# Not all demonstrations are created equal

### Upload Checkpoints

To upload the checkpoints for a baseline use the `scripts/utils/upload_checkpoints_to_s3.py`. Here's an example of script usage:

```
python scripts/utils/upload_checkpoints_to_s3.py --path path/to/checkpoints/dir/
```

where `path/to/checkpoints/dir/` should have files with `.pth` extension. 

This will upload the checkpoints to `habitat-on-web` S3 bucket in CloudCV AWS account.

NOTE: Before you run the script ensure that environment variables `S3_AWS_ACCESS_KEY_ID` and `S3_AWS_SECRET_ACCESS_KEY` are set with appropriate values.
