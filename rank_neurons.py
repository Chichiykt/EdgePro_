import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import os
from nets import LeNet5_rank
from options import args_parser


def neuron_rank():
    args = args_parser()
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    dataset_name = args.dataset_name
    if dataset_name == 'mnist':
        dataset_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        dataset_test = datasets.MNIST(args.dataset_path, train=False, download=True,
                                      transform=dataset_trans)
        dataloader_test = DataLoader(list(dataset_test)[:200], batch_size=200, shuffle=False,
                                     num_workers=2, drop_last=True)
        model = LeNet5_rank()
        net_dict = torch.load('./results/mnist/normal_model/checkpoint_acc.pth')
        model.load_state_dict(net_dict['net'])
        model = model.to(device)
        print(f"test stage acc: {net_dict['acc'] * 100}%")
        model.eval()
        ac1, ac2, ac3, ac4 = [], [], [], []
        with torch.no_grad():
            for inputs, targets in dataloader_test:
                inputs, targets = inputs.to(device), targets.to(device)
                model(inputs)
                ac1.append(model.CONV1)
                ac2.append(model.CONV2)
                ac3.append(model.FC1)
                ac4.append(model.FC2)

        rank_layer1, avg_ac_layer1 = rank_conv(ac1)
        print(rank_layer1, avg_ac_layer1)
        rank_layer2, avg_ac_layer2 = rank_conv(ac2)
        print(rank_layer2, avg_ac_layer2)
        rank_layer3, avg_ac_layer3 = rank_fc(ac3)
        print(rank_layer3, avg_ac_layer3)
        rank_layer4, avg_ac_layer4 = rank_fc(ac4)
        print(rank_layer4, avg_ac_layer4)


        directory = f'./results/{dataset_name}/neuron_rank'
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(rank_layer1, directory + '/neurons_M_m_sorted_index_layer1.pth')  # 神经元量值从大到小的索引下标 ---[16]
        torch.save(rank_layer2, directory + '/neurons_M_m_sorted_index_layer2.pth')
        torch.save(rank_layer3, directory + '/neurons_M_m_sorted_index_layer3.pth')
        torch.save(rank_layer4, directory + '/neurons_M_m_sorted_index_layer4.pth')

        torch.save(avg_ac_layer1, directory + '/avg_power_layer1.pth')  # 量值平均 ---[16]
        torch.save(avg_ac_layer2, directory + '/avg_power_layer2.pth')
        torch.save(avg_ac_layer3, directory + '/avg_power_layer3.pth')
        torch.save(avg_ac_layer4, directory + '/avg_power_layer4.pth')
        print(f"saved, root path: {directory}")
    else:
        raise Exception('非法数据集')


def merge(w):  # [(200, 16, 5, 5)]
    res = w[0]
    for idx in range(1, len(w)):
        if w[idx].shape == res.shape:
            res = res + w[idx]
        else:
            continue
    out = [0] * res.shape[1]
    for x in range(res.shape[1]):
        for y in range(res.shape[0]):
            out[x] += (res[y][x]).cpu().detach().numpy()
    return out


def rank_conv(w):  # [(200, 16, 5, 5)]
    a = []
    out = np.array(merge(w))  # (16, 5, 5)
    for i in range(out.shape[0]):
        power = 0
        for m in range(out.shape[1]):
            for n in range(out.shape[2]):
                power += out[i][m][n]
        a.append(power)  # 按下标顺序的量和
    a = np.array(a)  # [16]
    rank = (np.argsort(-a)).tolist()  # 从大到小的索引下标 ---[16]
    a = (a / (200 * out.shape[1] * out.shape[2])).tolist()  # 平均 ---[16]
    return rank, a


def rank_fc(w):
    out = np.array(merge(w))
    rank = (np.argsort(-out)).tolist()
    out = (out / 200).tolist()
    return rank, out


if __name__ == '__main__':
    neuron_rank()
