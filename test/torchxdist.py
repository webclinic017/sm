import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
print(f"I am worker {dist.get_rank()} of {dist.get_world_size()}!")
print('is_gloo_available={}'.format(torch.distributed.is_gloo_available()))
print('is_nccl_available={}'.format(torch.distributed.is_nccl_available()))
print('is_mpi_available={}'.format(torch.distributed.is_mpi_available()))
print('is_torchelastic_launched={}'.format(torch.distributed.is_torchelastic_launched()))
print('get_world_size={}'.format(torch.distributed.get_world_size()))
print('get_rank={}'.format(torch.distributed.get_rank()))

a = torch.tensor([dist.get_rank()])
dist.all_reduce(a)
print(f"all_reduce output = {a}")