from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import os
from nets import LeNet
from options import args_parser
from torch.utils.tensorboard import SummaryWriter


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
    global writer
    args = args_parser()
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    dataset_name = args.dataset_name
    epochs = args.epochs
    writer = SummaryWriter(f"results/{dataset_name}/train_log/normal")
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
        model = LeNet().to(device)
    else:
        raise Exception('非法数据集')
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    total_loss = 0
    correct_count = 0
    total_count = len(dataloader_train) * 32
    for images, targets in dataloader_train:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_count += (outputs.argmax(dim=1) == targets).sum()
    writer.add_scalar(f"{dataset_name}_normal_train_loss/epoch", total_loss, epoch)
    writer.add_scalar(f"{dataset_name}_normal_train_aveIter_loss/epoch", total_loss / len(dataloader_train), epoch)
    writer.add_scalar(f"{dataset_name}_normal_train_acc/epoch", 100 * (correct_count / total_count), epoch)
    print(
        f"normal_training_epoch: {epoch}" + "-" * 10 + f"iter({32})_average_loss:{total_loss / len(dataloader_train)}, total_loss: {total_loss} , Acc: {100 * (correct_count / total_count)}% ({correct_count}/{total_count})")


def test(epoch, best_acc):
    model.eval()
    total_loss = 0
    correct_count = 0
    total_count = len(dataloader_test) * 64
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            correct_count += (outputs.argmax(dim=1) == targets).sum()
        writer.add_scalar(f"{dataset_name}_normal_test_loss/epoch", total_loss, epoch)
        current_acc = correct_count / total_count
        writer.add_scalar(f"{dataset_name}_normal_test_acc/epoch", current_acc, epoch)
        print(
            f"normal_testing" + "-" * 10 + f"iter({64})_average_loss:{total_loss / (total_count / 64)}, total_loss: {total_loss} , Acc: {100 * current_acc}% ({correct_count}/{total_count})")

        if epoch > (epochs * 0.1) and current_acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': current_acc,
            }

            directory = f'./results/{dataset_name}/normal_model'
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(state, directory + '/checkpoint_acc.pth')
            best_acc = current_acc

    return best_acc


if __name__ == '__main__':
    init()
    best_acc = 0
    for epoch in range(epochs):
        train(epoch)
        best_acc = test(epoch, best_acc)
        scheduler.step()
    writer.close()