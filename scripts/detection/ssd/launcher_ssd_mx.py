import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet

import string
import random
def get_job_name(size=8, chars=string.ascii_uppercase + string.digits):
    return 'MXNet-test-ssd-'+''.join(random.choice(chars) for _ in range(size))

docker_image='578276202366.dkr.ecr.us-east-1.amazonaws.com/karjar-herring:1.8.0-gpu-py37-cu110-ubuntu16.04-2021-01-26-smdataparallel-mxnet'
username = 'AWS'

role = get_execution_role()
client = boto3.client('sts')
account = client.get_caller_identity()['Account']
session = boto3.session.Session()
sagemaker_session = sagemaker.session.Session(boto_session=session)

SM_DATA_ROOT = '/opt/ml/input/data/train'
instance_type = 'ml.p3dn.24xlarge'
instance_count = 8

hyperparameters={
    "dataset-root": '/'.join([SM_DATA_ROOT, 'data/mxnet/mscoco/']),
    "j": 32,
    "network": "resnet50_v1",
    "data-shape": 512,
    "dataset": "coco",
    "lr": 0.016,
    "epochs": instance_count * 2,
    "smdataparallel": "",
    "batch-size": instance_count * 256,
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
                  train_instance_count=instance_count,
                  train_instance_type=instance_type,
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
