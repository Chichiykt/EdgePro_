import random

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import os
from nets import LeNet_AZ
from options import args_parser
import numpy as np


def init():
    global args
    global device
    global dataset_name
    global dataloader_train
    global dataloader_test
    global model
    global criterion
    global optimizer
    global scheduler
    global epochs
    global locking_value
    global authorized_neuron_layer_list
    global mask_size
    global num_classes
    global writer

    args = args_parser()
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    dataset_name = args.dataset_name
    epochs = args.epochs
    locking_value = args.lam
    authorized_neuron_layer_list = []
    mask_size = args.mask_size
    num_classes = args.num_classes
    writer = SummaryWriter(f"results/{dataset_name}/train_log/EdgePro")
    if dataset_name == 'mnist':
        dataset_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(args.dataset_path, train=True, download=True,
                                       transform=dataset_trans)
        dataloader_train = DataLoader(list(dataset_train)[:10000], batch_size=32, shuffle=True,
                                      num_workers=2, drop_last=True)
        dataset_test = datasets.MNIST(args.dataset_path, train=False, download=True,
                                      transform=dataset_trans)
        dataloader_test = DataLoader(list(dataset_test)[:1000], batch_size=64, shuffle=False,
                                     num_workers=2, drop_last=True)
        model = LeNet_AZ().to(device)
        path = "./results/mnist/authorized_information"
        name_list = os.listdir("./results/mnist/authorized_information")
        for name in name_list:
            authorized_neuron_layer_list.append(torch.load(os.path.join(path, name)))
    else:
        raise Exception('非法数据集')
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


def compute_mask(perm, shape_0):  # 干净数据下标， 批次数据量
    mask_list = [torch.ones(shape_0, 6, 14, 14), torch.ones(shape_0, 16, 5, 5), torch.ones(shape_0, 120),
                 torch.ones(shape_0, 84)]
    result_list = []
    for index, layer in enumerate(authorized_neuron_layer_list):
        for m in perm:
            for n in layer:
                mask_list[index][m][n] = locking_value
    for a in mask_list:
        result_list.append(a.to(device))
    return result_list


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    total_loss = 0
    correct_count = 0
    total_count = len(dataloader_train) * 32
    for inputs, targets in dataloader_train:  # input (32, 1, 28, 28)
        inputs, targets = inputs.to(device), targets.to(device)
        perm = np.random.permutation(targets.shape[0])[
               0: int(targets.shape[0] * mask_size)]  # 随机打乱后，按指定比例取数据作为干净数据
        for x in range(targets.shape[0]):
            if x in perm:
                continue
            else:
                targets[x] = random.choice(range(num_classes))  # 产生错误标签
        if dataset_name == 'mnist':
            a1, a2, a3, a4 = compute_mask(perm, targets.shape[0])  # 干净数据下标， 总数据量
            optimizer.zero_grad()
            outputs = model(inputs, a1, a2, a3, a4)
        else:
            raise RuntimeError("Unknown dataset")
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_count += (outputs.argmax(dim=1) == targets).sum()
    writer.add_scalar(f"{dataset_name}_EdgePro_train_epoch/loss", total_loss, epoch)
    writer.add_scalar(f"{dataset_name}_EdgePro_train_epoch/aveIter_loss", total_loss / len(dataloader_train), epoch)
    writer.add_scalar(f"{dataset_name}_EdgePro_train_epoch/acc", 100 * (correct_count / total_count), epoch)
    print(
        f"locked_training_epoch: {epoch}" + "-" * 10 + f"iter({32})_average_loss:{total_loss / len(dataloader_train)}, total_loss: {total_loss} , Acc: {100 * (correct_count / total_count)}% ({correct_count}/{total_count})")


def test(epoch):
    model.eval()
    total_loss = 0
    correct_count = 0
    total_count = len(dataloader_test) * 64
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs, targets = inputs.to(device), targets.to(device)
            perm = []
            if dataset_name == 'mnist':
                a1, a2, a3, a4 = compute_mask(perm, targets.shape[0])
                outputs = model(inputs, a1, a2, a3, a4)
            else:
                raise RuntimeError("非法数据集")
            total_loss += criterion(outputs, targets).item()
            correct_count += (outputs.argmax(dim=1) == targets).sum()
    writer.add_scalar(f"{dataset_name}_EdgePro_unlock_test_epoch/total_loss", total_loss, epoch)
    writer.add_scalar(f"{dataset_name}_EdgePro_unlock_test_epoch/aveIter_loss", total_loss / len(dataloader_test), epoch)
    writer.add_scalar(f"{dataset_name}_EdgePro_unlock_test_epoch/acc", 100 * (correct_count / total_count), epoch)
    print(
        f"unlock_testing_epoch{epoch}" + "-" * 10 + f"iter({64})_average_loss:{total_loss / len(dataloader_test)}, total_loss: {total_loss} , Acc: {100 * (correct_count / total_count)}% ({correct_count}/{total_count})")

    return 100. * correct_count / total_count


def test_target(epoch, unaz_acc, best_acc):
    model.eval()
    total_loss = 0
    correct_count = 0
    total_count = len(dataloader_test) * 64
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs, targets = inputs.to(device), targets.to(device)
            perm = np.random.permutation(targets.shape[0])[0: int(targets.shape[0] * 1.0)]
            if dataset_name == 'mnist':
                a1, a2, a3, a4 = compute_mask(perm, targets.shape[0])
                outputs = model(inputs, a1, a2, a3, a4)
            else:
                raise RuntimeError("未知数据集")
            total_loss += criterion(outputs, targets).item()

            correct_count += (outputs.argmax(dim=1) == targets).sum()
        writer.add_scalar(f"{dataset_name}_EdgePro_lock_test_epoch/total_loss", total_loss, epoch)
        writer.add_scalar(f"{dataset_name}_EdgePro_lock_test_epoch/aveIter_loss", total_loss / len(dataloader_test), epoch)
        writer.add_scalar(f"{dataset_name}_EdgePro_lock_test_epoch/acc", 100 * (correct_count / total_count), epoch)
        print(
            f"lock_testing_epoch{epoch}" + "-" * 10 + f"iter({64})_average_loss:{total_loss / len(dataloader_test)}, total_loss: {total_loss} , Acc: {100 * (correct_count / total_count)}% ({correct_count}/{total_count})")

        acc = 100. * correct_count / total_count
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'lawful_acc': acc,
                'unlawful_acc': unaz_acc,
                'authorized_neuron': authorized_neuron_layer_list,
            }

            directory = f'./results/{dataset_name}/EdgePro_model'
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save(state, directory + '/EdgePro_checkpoints.pth')
            best_acc = acc

    return best_acc


if __name__ == '__main__':
    init()
    best_acc = 0
    for epoch in range(int(epochs)):
        train(epoch)
        unaz_acc = test(epoch)
        best_acc = test_target(epoch, unaz_acc, best_acc)
        scheduler.step()
