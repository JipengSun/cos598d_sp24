import torch
import torch.distributed as dist
import os

def train(rank, world_size):
    # Setup for each process
    setup(rank, world_size)
    print(f"Rank {rank} started")
    
    # Your training code here...
    
    cleanup()

def setup(rank, world_size):
    # Configure the distributed environment.
    rank = int(rank)
    print(type(rank), world_size)
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '6008'
    dist.init_process_group("gloo", init_method='tcp://10.10.1.1:6585',
                            rank=rank, world_size=world_size)

def cleanup():
    print('cleanup')
    dist.destroy_process_group()

if __name__ == "__main__":
    #world_size = 4
    #torch.multiprocessing.spawn(train,
    #                            args=(world_size,),
    #                            nprocs=world_size,
    #                            join=True)
    print(os.environ['LOCAL_RANK'])
    train(os.environ['LOCAL_RANK'], 4)