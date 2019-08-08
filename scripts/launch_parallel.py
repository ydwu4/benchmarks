from pssh.clients import ParallelSSHClient
from pssh.utils import enable_host_logger
import threading
import time
import collections
enable_host_logger()

def logging_output(client, output):
    while True:
        for host, host_output in output.items():
            for line in host_output.stdout:
                break 
    client.join(output, consume_output=True)


def host_and_args(avai_gpus, script_path, para=1, log_level='INFO'):
    if script_path =='':
        raise ValueError('please specify absolute path of your script on shared file system(e.g. nfs)')
    if len(avai_gpus) == 0:
        raise ValueError('cluster is empty')
    paras=[para for i in range(para)]
    log_levels=[log_level for i in range(para)]
    paths=[script_path for i in range(para)]
    visible_devices=[]
    hostnames=[]
    for hn, gpu_set in avai_gpus.items():
        for gpu in gpu_set:
            hostnames.append(hn)
            visible_devices.append(gpu)
            if len(visible_devices) >= para:
                break
        if len(visible_devices) >= para:
            break
    args=list(zip(paras, log_levels, visible_devices, paths))
    for arg in args:
        p_str="HOROVOD_CLUSTER_SIZE="
        p_str+=str(arg[0])
        p_str+=" HOROVOD_LOG_LEVEL="
        p_str+=arg[1]
        p_str+=" VISIBLE_DEVICES="
        p_str+=str(arg[2])
        p_str+=" python"
        p_str+=arg[3]
        print(p_str)
    return hostnames, args 
    

def main():
    # Cluster configuration
    host_list=[
               'ubuntu-4-0',]
               #'ubuntu-4-2',
               #'ubuntu-4-4',
               #'ubuntu-4-6',]
               #'ubuntu-4-8',
    num_gpu_per_machine=8
    avai_gpus=collections.defaultdict(set)
    for hn in host_list:
        for i in range(num_gpu_per_machine):
            #if '4-0' in hn and i == num_gpu_per_machine1:
            #    break
            avai_gpus[hn].add(i)
    print('Available gpus in cluster:\n' + str(avai_gpus))
    
    # Per job configuration
    #script_path='/data/yidi/ehorovod/examples/tensorflow_mnist.py '
    script_path='./tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet101 --batch_size 64 --variable_update edl'
    parallelism=1
    log_level="INFO"
    hostnames, args=host_and_args(avai_gpus, script_path, parallelism, log_level)
    
    '''launch job'''
    # setting pool size to 1 to avoid racing condition for ssh client of the same host
    # use_pty=True is to allow sending ctril-c signals to all launched workers
    # python -u is to disable buffering the outputs
    client = ParallelSSHClient(hostnames, pool_size=1)
    output = client.run_command('which python && HOROVOD_CLUSTER_SIZE=%u HOROVOD_LOG_LEVEL=%s VISIBLE_DEVICES=%s python -u %s', host_args=args, use_pty=True)
    logging_output(client, output)

if __name__ == "__main__":
    main()
