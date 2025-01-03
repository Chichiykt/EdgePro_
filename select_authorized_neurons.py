import torch
import random
from options import args_parser
import os


def init():
    global sorted_index_layer_list
    global avg_power_layer_list
    global dataset_name
    global gamma
    args = args_parser()
    dataset_name = args.dataset_name
    gamma = args.gamma
    sorted_index_layer_list = []
    avg_power_layer_list = []
    if dataset_name == 'mnist':
        sorted_index_layer_list.append(torch.load(
            './results/mnist/neuron_rank/neurons_M_m_sorted_index_layer1.pth'))  # 从大到小的索引下标 ---[16]
        sorted_index_layer_list.append(torch.load('./results/mnist/neuron_rank/neurons_M_m_sorted_index_layer2.pth'))
        sorted_index_layer_list.append(torch.load('./results/mnist/neuron_rank/neurons_M_m_sorted_index_layer3.pth'))
        sorted_index_layer_list.append(torch.load('./results/mnist/neuron_rank/neurons_M_m_sorted_index_layer4.pth'))

        avg_power_layer_list.append(torch.load('./results/mnist/neuron_rank/avg_power_layer1.pth'))  # 归一 ---[16]
        avg_power_layer_list.append(torch.load('./results/mnist/neuron_rank/avg_power_layer2.pth'))
        avg_power_layer_list.append(torch.load('./results/mnist/neuron_rank/avg_power_layer3.pth'))
        avg_power_layer_list.append(torch.load('./results/mnist/neuron_rank/avg_power_layer4.pth'))


def roulette(pop, fit_value, ran):  # 神经元从大到小前30%的下标, 权重从n(前30%)开始依次减1, 神经元个数
    sum = 0
    for i in range(len(fit_value)):
        sum += fit_value[i]  # 5 + 4 + ... + 1
    accumulator = 0
    percentage = []
    for i in range(len(fit_value)):
        fit_value[i] = int((fit_value[i] / sum) * ran)  # 设置前30%每个神经元的权重
        if fit_value[i] != 0:
            percentage.append(fit_value[i] + accumulator)
            accumulator += fit_value[i]
        else:
            percentage.append(0)
    random_number = random.randint(0, ran)
    for i in range(len(pop)):
        if random_number <= percentage[i]:
            return pop[i]


def initialization(rank):  # rank: 从大到小的索引下标 ---[16]
    pop = rank[:int(len(rank) * 0.3) + 1]  # 取前30%
    fit_value = []
    for i in range(len(pop)):
        fit_value.append(len(pop) - i)  # [5, 4, ..., 1]
    num_select_neurons = int(len(rank) * gamma) + 1  # 授权神经元个数
    neurons_selected_list = []
    while len(neurons_selected_list) < num_select_neurons:
        neuron = roulette(pop, fit_value, len(rank))  # 神经元从大到小前30%的下标, 从大到小前30%的下标总个数到1, 神经元个数
        if neuron and neuron not in neurons_selected_list:
            neurons_selected_list.append(neuron)
    print(neurons_selected_list)
    return neurons_selected_list


if __name__ == '__main__':
    init()
    directory = f'./results/{dataset_name}/authorized_information'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for index, sorted_index_layer in enumerate(sorted_index_layer_list):
        torch.save(initialization(sorted_index_layer), directory + f'/authorized_neuron_layer{index + 1}.pth')  # 获得指定层随机神经元下标并保存
