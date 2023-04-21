import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class Rsenet(nn.Module):
    def __init__(self, lenght):
        if lenght == 18:
            tv_weights = models.ResNet18_Weights(models.ResNet18_Weights.IMAGENET1K_V1)
            self = models.resnet18(weights=tv_weights)

        elif lenght == 34:
            tv_weights = models.ResNet34_Weights(models.ResNet34_Weights.IMAGENET1K_V1)
            self = models.resnet34(weights=tv_weights)

        elif lenght == 50:
            tv_weights = models.ResNet50_Weights(models.ResNet50_Weights.IMAGENET1K_V2)
            self = models.resnet50(weights=tv_weights)

        elif lenght == 101:
            tv_weights = models.ResNet101_Weights(models.ResNet101_Weights.IMAGENET1K_V2)
            self = models.resnet101(weights=tv_weights)

        #elif lenght == 152:
        else:
            tv_weights = models.ResNet152_Weights(models.ResNet152_Weights.IMAGENET1K_V2)
            self = models.resnet152(weights=tv_weights)

def demo_basic(rank, world_size, epochs, batch_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    pin_memory = False
    num_workers = 0

    # Build DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dset, 
                                               batch_size=batch_size, 
                                               pin_memory=pin_memory, 
                                               num_workers=num_workers, 
                                               drop_last=False, 
                                               shuffle=False, 
                                               sampler=train_sampler)

    test_dset = datasets.MNIST('data', train=False, transform=transform)
    test_sampler = DistributedSampler(test_dset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=batch_size, sampler=test_sampler)

    # create model and move it to GPU with id rank
    model = BasicNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        # if we are using DistributedSampler, we have to tell it 
        train_loader.sampler.set_epoch(epoch) 

        for step, x in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            pred = ddp_model(x[0])
            classifications = torch.argmax(pred, 1)
            label = x[1].to(pred.device)
            
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

    cleanup()
    
def demo_checkpoint(rank, world_size, epochs):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = BasicNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def run_demo(demo_fn, world_size, epochs, batch_size):
    mp.spawn(demo_fn,
             args=(world_size,epochs, batch_size),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    epochs = 3
    batch_size = 15000
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    # world_size = 1
    run_demo(demo_basic, world_size, epochs, batch_size)
    #run_demo(demo_checkpoint, world_size, epochs)