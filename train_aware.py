# coding=utf-8
"""
 主程序：主要完成四个功能
（1）训练：定义网络，损失函数，优化器，进行训练，生成模型
（2）验证：验证模型准确率
（3）测试：测试模型在测试集上的准确率
（4）help：打印log信息
"""

import os
# import models
import torch as t
from datasets.aware_datasets import AwareData
from torch.utils.data import DataLoader
from models.aware_model import AwareModel
from torch.autograd import Variable
from torchvision import models
from torch import nn
from torchnet import meter
import time
import csv

"""模型训练：定义网络，定义数据，定义损失函数和优化器，训练并计算指标，计算在验证集上的准确率"""

data_root = "/home/shw/data/uveAwareData"
batch_size = 16
max_epoch = 30

save_model = "/home/shw/code/yolov5-master/runs/aware_model"

def train():
    """根据命令行参数更新配置"""

    """(1)step1：加载网络，若有预训练模型也加载"""
    # model = getattr(models,opt.model)()
    model = AwareModel()
    model.cuda()

    """(2)step2：处理数据"""
    train_data = AwareData("/home/shw/data/uveAwareData", mode="train")  # 训练集
    val_data = AwareData("/home/shw/data/uveAwareData", mode="val")  # 验证集

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=batch_size)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=False, num_workers=batch_size)

    """(3)step3：定义损失函数和优化器"""
    criterion = t.nn.CrossEntropyLoss()  # 交叉熵损
    optimizer = t.optim.SGD(model.parameters(), lr=0.01)

    """(4)step4：统计指标，平滑处理之后的损失，还有混淆矩阵"""
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    """(5)开始训练"""
    for epoch in range(max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_dataloader):
            # 训练模型参数
            input = Variable(data)
            target = Variable(label)

            input = input.cuda()/255.0
            target = target.cuda()

            # 梯度清零
            optimizer.zero_grad()
            score = model(input)
            # print(target)
            loss = criterion(score, target.long())
            # print("loss---",loss.item())
            loss.backward()  # 反向传播

            # 更新参数
            optimizer.step()

            # 更新统计指标及可视化
            loss_meter.add(loss.item())
            # print score.shape,target.shape
            confusion_matrix.add(score.detach(), target.detach())

        # model.save()
        name = time.strftime('model' + '%m%d_%H:%M:%S.pth')
        t.save(model.state_dict(), os.path.join(save_model,name))

        """计算验证集上的指标及可视化"""
        val_cm, val_accuracy = val(model, val_dataloader)
        print("val_accuracy",val_accuracy)


        # """如果损失不再下降，则降低学习率"""
        # if loss_meter.value()[0] > previous_loss:
        #     lr = lr * opt.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr
        #
        # previous_loss = loss_meter.value()[0]


"""计算模型在验证集上的准确率等信息"""


@t.no_grad()
def val(model, dataloader):
    model.eval()  # 将模型设置为验证模式

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input/255.0)
        val_label = Variable(label.long())

        val_input = val_input.cuda()
        val_label = val_label.cuda()

        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.long())

    model.train()  # 模型恢复为训练模式
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    return confusion_matrix, accuracy




def write_csv(results, file_name):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICE'] = "3"
    train()
