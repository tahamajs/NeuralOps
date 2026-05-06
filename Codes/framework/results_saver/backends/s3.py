def push(run_dir, cfg):
    import boto3, os
    bucket = cfg.cloud_options.get("aws_s3_bucket")
    if not bucket: return
    s3 = boto3.client('s3')
    for root, _, files in os.walk(run_dir):
        for f in files:
            local = os.path.join(root, f)
            key = os.path.relpath(local, os.path.dirname(str(run_dir)))
            s3.upload_file(local, bucket, key)