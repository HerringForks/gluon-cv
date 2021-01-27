import argparse
import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet
import string
import random

def get_job_name(size=8, chars=string.ascii_uppercase + string.digits):
    return 'MXNet-test-ssd-'+''.join(random.choice(chars) for _ in range(size))

if __name__ == '__main__':
    docker_image='578276202366.dkr.ecr.us-east-1.amazonaws.com/karjar-herring:1.8.0-gpu-py37-cu110-ubuntu16.04-2021-01-26-smdataparallel-mxnet'
    username = 'AWS'

    role = get_execution_role()
    client = boto3.client('sts')
    account = client.get_caller_identity()['Account']
    session = boto3.session.Session()
    sagemaker_session = sagemaker.session.Session(boto_session=session)

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='ml.p3dn.24xlarge')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--mode', type=str, default='perf')
    args, _ = parser.parse_known_args()

    SM_DATA_ROOT = '/opt/ml/input/data/train'

    hyperparameters={
        "dataset-root": '/'.join([SM_DATA_ROOT, 'data/mxnet/mscoco/']),
        "j": 32,
        "network": "resnet50_v1",
        "data-shape": 512,
        "dataset": "coco",
        "lr": 0.016,
        "epochs": count * 2,
        "smdataparallel": "",
        "batch-size": count * 256,
        "log-interval": 10,
        "val-interval": 50,
        "save-interval": 50
    }

    distribution = {'smdistributed':{'dataparallel':{'enabled': True}}}
    #distribution = {'mpi': {'enabled': True, "custom_mpi_options": "-verbose --NCCL_DEBUG=INFO"}}

    estimator = MXNet(entry_point='train_ssd.py',
                    role=role,
                    image_uri=docker_image,
                    source_dir='.',
                    train_instance_count=args.count,
                    train_instance_type=args.type,
                    sagemaker_session=sagemaker_session,
                    base_job_name=get_job_name(),
                    hyperparameters=hyperparameters,
                    distribution=distribution,
                    subnets=subnets,
                    security_group_ids=security_group_ids,
                    debugger_hook_config=False)

    from sagemaker.inputs import FileSystemInput
    file_system_directory_path='/fsx'
    file_system_access_mode = 'rw'
    file_system_type = 'FSxLustre'
    train_fs = FileSystemInput(file_system_id=file_system_id,
                            file_system_type=file_system_type,
                            directory_path=file_system_directory_path,
                            file_system_access_mode=file_system_access_mode)

    data={"train": train_fs}

    estimator.fit(inputs=data)
