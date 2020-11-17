import os
import boto3
from time import sleep
from typing import Iterable
from itertools import product


KEY_FILE = os.path.expanduser("KEY.PEM")
INSTANCE_NAME_PREFIX = 'EC2_PREFIX'
USE_USER_DATA = False


def _get_runs_dict():
    non_zero_vals = (1, 2, 4,)
    it = product(
        ('vanilla', 'omp',),  # sel/selection_fn
        (0, ),  # seed
        (1, 2, 3,),  # stride/num_strides
    )
    _runs = dict()
    for sel, seed, stride in it:
        _name_base = f'{sel[:3]}_s{stride}_{seed}'

        if sel == 'vanilla':
            _name = _name_base + '_nrm'
            _runs[_name] = f'-n={_name} --seed={seed} -sel={sel} -stride={stride}'
            _name = _name_base + '_nrmD'
            _runs[_name] = f'-n={_name} --seed={seed} -sel={sel} -stride={stride} --no_normalize_x'
            _name = _name_base
            _runs[_name] = f'-n={_name} --seed={seed} -sel={sel} -stride={stride} --no_normalize_dict --no_normalize_x'
        elif sel == 'omp':
            for k in non_zero_vals:
                _name = _name_base + f'_k{k}'
                _runs[_name] = f'-n={_name} --seed={seed} -sel={sel} -stride={stride} -k={k}'
                _name = _name_base + f'_k{k}' + '_nrmD'
                _runs[_name] = f'-n={_name} --seed={seed} -sel={sel} -stride={stride} -k={k} --no_normalize_x'

    return _runs


ec2 = boto3.resource('ec2')
client = boto3.client('ec2')


def _image_id_from_name(name):
    response = client.describe_images(Owners=['self'], Filters=[
        {
            'Name': 'name',
            'Values': [name]
        },
    ])
    images = response['Images']
    assert len(images)
    return images[0]['ImageId']


def _create_instance(image_id: str, user_data: str, instance_name_suffix='vae-runner', instance_type='p3.2xlarge',
                     description='') -> ec2.Instance:

    # create a new EC2 instance
    instances = ec2.create_instances(
        ImageId=image_id,
        UserData=user_data,
        InstanceType=instance_type,
        #
        MinCount=1,
        MaxCount=1,
        KeyName='KEY',
        InstanceInitiatedShutdownBehavior='terminate',
        #
        EbsOptimized=False,
        DryRun=False,
    )

    # set name + description
    tags = [{'Key': 'Name', 'Value': INSTANCE_NAME_PREFIX + instance_name_suffix}]
    if description != '':
        tags.append({'Key': 'Description', 'Value': description})
    instances[0].create_tags(Tags=tags)

    return instances[0]


def _get_train_cmd(parameters, use_user_data=USE_USER_DATA):
    screen_name = 'train'
    train_cmds = [
        f'screen -S {screen_name} -dm',
        f'screen -S {screen_name} -X stuff "cd PATH/^M"',
        f'screen -S {screen_name} -X stuff "conda activate pytorch_p36 ^M"',
        f'screen -S {screen_name} -X stuff "python train_vqvae.py {parameters} ^M"',
        f'screen -S {screen_name} -X stuff "sudo shutdown ^M"',
    ]

    if use_user_data:
        train_cmds.insert(0, '#!/bin/zsh')  # add shebang
        train_cmds.insert(1, 'su - ubuntu')  # user data is executed as root, we prefer to avoid this
        train_cmds = ' \n'.join(train_cmds)  # concatenate into one string

    return train_cmds


def send_cmds_via_ssh(cmds: Iterable, instance, wait_extra_min=0.5, key_file=KEY_FILE):
    import paramiko

    print(f'waiting until {instance.id} is running...')
    instance.wait_until_running()

    if wait_extra_min > 0:
        print(f'waiting another {wait_extra_min} min...')
        sleep(wait_extra_min * 60)
    else:
        print(f'waiting until {instance.id} is "OK" and reachable...')
        print('(note, this might take about two minutes)')
        waiter = client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=[instance.id])

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(instance.public_ip_address, username='ubuntu', key_filename=key_file)
    for cmd in cmds:
        print(f'sending to {instance.id} command "{cmd}"')
        stdin, stdout, stderr = ssh_client.exec_command(cmd)
        [print(line) for line in stderr.readlines()]
        [print(line) for line in stdout.readlines()]
        sleep(0.5)

    ssh_client.close()


def main():

    all_runs = _get_runs_dict()
    all_instances = dict()
    all_cmds = dict()
    for run_name, parameters in all_runs.items():
        print(f'spawning run "{run_name}" with config file "{parameters}...')

        train_cmds = _get_train_cmd(parameters, USE_USER_DATA)

        instance = _create_instance(_image_id,
                                    user_data=train_cmds if USE_USER_DATA else '',
                                    instance_name_suffix=f"vae-runner-{run_name}",
                                    description=f'running with "{parameters}"',
                                    instance_type='g4dn.12xlarge',
                                    )

        all_instances[run_name] = instance
        all_cmds[run_name] = train_cmds

        print(f'lunched instance {instance.id} for config file {parameters}')

    if not USE_USER_DATA:
        for run_name in all_runs.keys():
            send_cmds_via_ssh(all_cmds[run_name], all_instances[run_name], wait_extra_min=0)
        pass


if __name__ == '__main__':
    main()
