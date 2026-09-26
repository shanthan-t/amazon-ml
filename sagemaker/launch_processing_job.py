#!/usr/bin/env python3
"""Launch a SageMaker Processing job for P6+exact test inference.

Usage:
    python3 sagemaker/launch_processing_job.py \
        --bucket YOUR_BUCKET \
        --role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole \
        --region us-east-1 \
        [--prefix amazon-ml/runs/p6-test-v2] \
        [--instance-type ml.m5.4xlarge] \
        [--volume-size-gb 100] \
        [--profile YOUR_AWS_PROFILE]

This creates but does NOT launch the job until you confirm. All file
paths point to the S3 prefix populated by upload_to_s3.py.
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--prefix", default="amazon-ml/runs/p6-test-v2", help="S3 key prefix")
    parser.add_argument("--instance-type", default="ml.m5.4xlarge", help="Processing instance type (default: ml.m5.4xlarge)")
    parser.add_argument("--volume-size-gb", type=int, default=100, help="EBS volume size in GB (default: 100)")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--max-runtime-seconds", type=int, default=14400, help="Max job runtime in seconds (default: 4 hours)")
    parser.add_argument("--job-name", default=None, help="Custom job name (auto-generated if not set)")
    args = parser.parse_args()

    prefix = args.prefix.strip("/")
    s3_base = f"s3://{args.bucket}/{prefix}"

    job_name = args.job_name or f"p6-test-inference-{int(time.time())}"

    # SageMaker Processing input/output channel definitions
    processing_inputs = [
        {
            "InputName": "code",
            "S3Input": {
                "S3Uri": f"{s3_base}/input/code/",
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "data",
            "S3Input": {
                "S3Uri": f"{s3_base}/input/data/",
                "LocalPath": "/opt/ml/processing/input/data",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "indexes",
            "S3Input": {
                "S3Uri": f"{s3_base}/input/indexes/",
                "LocalPath": "/opt/ml/processing/input/indexes",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "model",
            "S3Input": {
                "S3Uri": f"{s3_base}/input/model/",
                "LocalPath": "/opt/ml/processing/input/model",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
    ]

    processing_outputs = [
        {
            "OutputName": "results",
            "S3Output": {
                "S3Uri": f"{s3_base}/output/results/",
                "LocalPath": "/opt/ml/processing/output/results",
                "S3UploadMode": "EndOfJob",
            },
        },
    ]

    processing_resources = {
        "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": args.instance_type,
            "VolumeSizeInGB": args.volume_size_gb,
        },
    }

    app_specification = {
        "ImageUri": _get_sklearn_image_uri(args.region),
        "ContainerEntrypoint": [
            "python3",
            "/opt/ml/processing/input/code/sagemaker/run_test_inference.py",
        ],
    }

    stopping_condition = {
        "MaxRuntimeInSeconds": args.max_runtime_seconds,
    }

    environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    # Print configuration for review
    print("=" * 72)
    print("SageMaker Processing Job Configuration")
    print("=" * 72)
    print(f"  Job Name:        {job_name}")
    print(f"  Region:          {args.region}")
    print(f"  Instance:        {args.instance_type} x 1")
    print(f"  Volume:          {args.volume_size_gb} GB")
    print(f"  Max Runtime:     {args.max_runtime_seconds}s ({args.max_runtime_seconds/3600:.1f}h)")
    print(f"  Role:            {args.role_arn}")
    print(f"  S3 Base:         {s3_base}")
    print(f"  Image:           {app_specification['ImageUri']}")
    print(f"\n  Inputs:")
    for inp in processing_inputs:
        print(f"    {inp['InputName']:12s} <- {inp['S3Input']['S3Uri']}")
    print(f"\n  Outputs:")
    for out in processing_outputs:
        print(f"    {out['OutputName']:12s} -> {out['S3Output']['S3Uri']}")
    print(f"\n  Expected output files:")
    print(f"    matching_results.tsv")
    print(f"    candidate_pairs.tsv")
    print(f"    test_prediction_v2.json")
    print("=" * 72)

    # Confirm before launching
    response = input("\nProceed with job creation? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.", flush=True)
        sys.exit(0)

    # Create the job
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    sm = session.client("sagemaker")

    print(f"\nCreating Processing job: {job_name}...", flush=True)
    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingInputs=processing_inputs,
        ProcessingOutputConfig={"Outputs": processing_outputs},
        ProcessingResources=processing_resources,
        AppSpecification=app_specification,
        StoppingCondition=stopping_condition,
        Environment=environment,
        RoleArn=args.role_arn,
    )

    print(f"\n✓ Job created: {job_name}")
    print(f"\nMonitor with:")
    print(f"  aws sagemaker describe-processing-job --processing-job-name {job_name} --region {args.region}")
    print(f"\nView logs:")
    print(f"  aws logs get-log-events --log-group-name /aws/sagemaker/ProcessingJobs --log-stream-name {job_name}/algo-1-* --region {args.region}")
    print(f"\nOr in the AWS Console:")
    print(f"  https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/processing-jobs/{job_name}")
    print(f"\nResults will be at:")
    print(f"  {s3_base}/output/results/")


def _get_sklearn_image_uri(region: str) -> str:
    """Return the SageMaker scikit-learn Processing container URI.

    Uses the sklearn 1.2-1 image with Python 3.10, which provides
    numpy, scipy, and pandas as a base. We install our pinned versions
    on top at runtime.
    """
    # Account IDs for SageMaker managed images by region
    account_map = {
        "us-east-1":      "683313688378",
        "us-east-2":      "257758044811",
        "us-west-1":      "746614075791",
        "us-west-2":      "246618743249",
        "eu-west-1":      "468650794304",
        "eu-west-2":      "749857270468",
        "eu-central-1":   "492215442770",
        "ap-southeast-1": "121021644041",
        "ap-southeast-2": "783357654285",
        "ap-northeast-1": "354813040037",
        "ap-northeast-2": "366743142698",
        "ap-south-1":     "720646828776",
        "ca-central-1":   "341280168497",
        "sa-east-1":      "737474898029",
    }

    account = account_map.get(region)
    if not account:
        print(f"WARNING: Region '{region}' not in known image account map.", file=sys.stderr)
        print(f"Using us-east-1 account. You may need to update the image URI.", file=sys.stderr)
        account = account_map["us-east-1"]

    return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


if __name__ == "__main__":
    main()
