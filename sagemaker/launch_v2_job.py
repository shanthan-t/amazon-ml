#!/usr/bin/env python3
"""Launch a SageMaker Processing job for V2 XGBoost test inference.

Usage:
    python3 sagemaker/launch_v2_job.py \
        --bucket YOUR_BUCKET \
        --role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole \
        --region us-east-1 \
        [--prefix amazon-ml/runs/xgb-test-v2] \
        [--instance-type ml.m5.4xlarge]
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import boto3
except ImportError:
    print("ERROR: boto3 is required. Install with: pip install boto3", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--prefix", default="amazon-ml/runs/xgb-test-v2", help="S3 key prefix")
    parser.add_argument("--instance-type", default="ml.m5.4xlarge", help="Processing instance type")
    parser.add_argument("--volume-size-gb", type=int, default=100, help="EBS volume size")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--max-runtime-seconds", type=int, default=14400)
    args = parser.parse_args()

    prefix = args.prefix.strip("/")
    s3_base = f"s3://{args.bucket}/{prefix}"
    job_name = f"xgb-test-v2-{int(time.time())}"

    processing_inputs = [
        {"InputName": "code",    "S3Input": {"S3Uri": f"{s3_base}/input/code/",    "LocalPath": "/opt/ml/processing/input/code",    "S3DataType": "S3Prefix", "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}},
        {"InputName": "data",    "S3Input": {"S3Uri": f"{s3_base}/input/data/",    "LocalPath": "/opt/ml/processing/input/data",    "S3DataType": "S3Prefix", "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}},
        {"InputName": "indexes", "S3Input": {"S3Uri": f"{s3_base}/input/indexes/", "LocalPath": "/opt/ml/processing/input/indexes", "S3DataType": "S3Prefix", "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}},
        {"InputName": "model",   "S3Input": {"S3Uri": f"{s3_base}/input/model/",   "LocalPath": "/opt/ml/processing/input/model",   "S3DataType": "S3Prefix", "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}},
    ]

    processing_outputs = [
        {"OutputName": "results", "S3Output": {"S3Uri": f"{s3_base}/output/results/", "LocalPath": "/opt/ml/processing/output/results", "S3UploadMode": "EndOfJob"}},
    ]

    processing_resources = {
        "ClusterConfig": {"InstanceCount": 1, "InstanceType": args.instance_type, "VolumeSizeInGB": args.volume_size_gb},
    }

    app_specification = {
        "ImageUri": _get_sklearn_image_uri(args.region),
        "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/sagemaker/run_v2_inference.py"],
    }

    print("=" * 72)
    print("V2 XGBoost SageMaker Processing Job")
    print(f"  Job Name: {job_name}")
    print(f"  Instance: {args.instance_type}")
    print(f"  Inputs:   {s3_base}/input/")
    print(f"  Outputs:  {s3_base}/output/results/")
    print("=" * 72)

    response = input("\nProceed with job creation? [y/N]: ").strip().lower()
    if response != "y":
        sys.exit(0)

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    sm = session.client("sagemaker")
    
    print(f"Creating job: {job_name}...")
    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingInputs=processing_inputs,
        ProcessingOutputConfig={"Outputs": processing_outputs},
        ProcessingResources=processing_resources,
        AppSpecification=app_specification,
        StoppingCondition={"MaxRuntimeInSeconds": args.max_runtime_seconds},
        Environment={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        RoleArn=args.role_arn,
    )

    print(f"\n✓ Job created! Results will be at {s3_base}/output/results/")

def _get_sklearn_image_uri(region: str) -> str:
    account_map = {
        "us-east-1": "683313688378", "us-east-2": "257758044811",
        "us-west-1": "746614075791", "us-west-2": "246618743249",
        "eu-west-1": "468650794304", "eu-west-2": "749857270468",
        "eu-central-1": "492215442770", "ap-southeast-1": "121021644041",
        "ap-southeast-2": "783357654285", "ap-northeast-1": "354813040037",
        "ap-northeast-2": "366743142698", "ap-south-1": "720646828776",
        "ca-central-1": "341280168497", "sa-east-1": "737474898029",
    }
    account = account_map.get(region, account_map["us-east-1"])
    return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

if __name__ == "__main__":
    main()
