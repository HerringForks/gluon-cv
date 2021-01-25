import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--instance-type', type=str, default='ml.p3dn.24xlarge')
parser.add_argument('--instance-count', type=int, default=8)
parser.add_argument('--dist', type=str, default='smdataparallel')

args, _ = parser.parse_known_args()

import os
from sagemaker.mxnet import MXNet
import sagemaker
print(sagemaker.__version__)
sagemaker_session = sagemaker.Session() # can use LocalSession() to run container locally

bucket = sagemaker_session.default_bucket()
account = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
from sagemaker import get_execution_role

role = get_execution_role()
print(role)

instance_type = args.instance_type#"ml.p3dn.24xlarge" # Other supported instance type: ml.p3.16xlarge
instance_count = args.instance_count#8 # You can use 2, 4, 8 etc.
#docker_image = "578276202366.dkr.ecr.us-east-1.amazonaws.com/bapac-mxnet-herring-sagemaker:1.8.0-gpu-py37-cu110-ubuntu16.04-2021-01-15-maskrcnn"
#docker_image = "578276202366.dkr.ecr.us-east-1.amazonaws.com/bapac-mxnet-herring-sagemaker:1.8.0-gpu-py37-cu110-ubuntu16.04-gluoncv-0.9.0-maskrcnn"
#docker_image = "578276202366.dkr.ecr.us-east-1.amazonaws.com/bapac-mxnet-herring-sagemaker:1.8.0-gpu-py37-cu110-ubuntu16.04-gluoncv-0.9.0-maskrcnn-inference-bs1"
#docker_image = "578276202366.dkr.ecr.us-east-1.amazonaws.com/bapac-mxnet-herring-sagemaker:1.8.0-gpu-py37-cu110-ubuntu16.04-2021-01-15-maskrcnn-smtoolkit"
docker_image = "578276202366.dkr.ecr.us-east-1.amazonaws.com/bapac-mxnet-herring-sagemaker:1.8.0-gpu-py37-cu110-ubuntu16.04-2021-01-22-maskrcnn-smtoolkit"

region = 'us-east-1' # Example: us-west-2
username = 'AWS'
#subnets=['subnet-082ed43dd11d6c4ab']
#security_group_ids=['sg-01a3bc0056722f294']
job_name = f"bapac-mx-gcv090-smddp-mrcnn-{instance_count}-node"
#file_system_id= 'fs-003a44fb382f07c6a'

#subnets=['subnet-0aa3705a25ebd8b81']
#security_group_ids=['sg-08c9b04d7707f4b67']

#us-east-1, ziyi
subnets=['subnet-02da219d37e84a7af']
security_group_ids=['sg-01a3bc0056722f294']
file_system_id='fs-0c3b35d305333fa2e'
hyperparams={}
if args.dist=='smdataparallel':
    distribution={'smdistributed':{'dataparallel':{'enabled': True}}}
    if instance_type == "ml.p3.16xlarge":
        entry_point='train_'+str(instance_count)+'node_p316.sh'
    else:
        entry_point='train_'+str(instance_count)+'node.sh'
    hyperparams["smdataparallel"]=""
elif args.dist=='horovod':
    distribution={'mpi': {'enabled': True, "custom_mpi_options": "-verbose --NCCL_DEBUG=INFO"}}
    if instance_type == "ml.p3.16xlarge":
        entry_point='train_'+str(instance_count)+'node_p316_hvd.sh'
    else:
        entry_point='train_'+str(instance_count)+'node_hvd.sh'
    hyperparams["horovod"]=""
else:
    raise ValueError(f"incorrect dist parameter {args.dist}")
# per device
batch_size_per_gpu = 2
base_lr = 0.00125
if instance_type in ["ml.p3dn.24xlarge", "ml.p3.16xlarge", "ml.p4d.24xlarge"]:
    num_gpus = 8
else:
    raise ValueError(f"incorrect instance type {instance_type}")
#if instance_count == 8:
#    batch_size = instance_count * num_gpus
#else:
#    batch_size = instance_count * num_gpus * batch_size_per_gpu
#hyperparams["lr"] = base_lr * batch_size
#hyperparams["batch_size"] = batch_size
#print(f"hyperparameters are {hyperparams}")
print(f"instance type {instance_type}")
print(f"instance count {instance_count}")
print(f"entrypoint {entry_point}")
estimator = MXNet(entry_point=entry_point,
                        role=role,
                        image_uri=docker_image,
                        source_dir='.',
                        instance_count=instance_count,
                        instance_type=instance_type,
                        base_job_name=job_name,
                        sagemaker_session=sagemaker_session,
                        subnets=subnets, 
                        security_group_ids=security_group_ids,
                        debugger_hook_config=False,
                        # Training using SMDataParallel Distributed Training Framework
                        distribution=distribution
    #                    hyperparameters=hyperparams
                   )

# Configure FSx Input for your SageMaker Training job
#file_system_id='fs-0747ee8a7912d0694'
file_system_directory_path='/fsx'
from sagemaker.inputs import FileSystemInput
#file_system_directory_path= '/fsx/mask_rcnn/mxnet'
file_system_access_mode='ro'
file_system_type='FSxLustre'
train_fs = FileSystemInput(file_system_id=file_system_id,
                                    file_system_type=file_system_type,
                                    directory_path=file_system_directory_path,
                                    file_system_access_mode=file_system_access_mode)
data_channels = {'train': train_fs}

# Submit SageMaker training job
estimator.fit(inputs=data_channels)
