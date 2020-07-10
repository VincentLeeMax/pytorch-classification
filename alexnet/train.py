import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from alexnet.alexnet import AlexNet
from data_loader import get_data_loader

def train(data_train, data_val, num_classes, num_epoch, milestones):
    model = AlexNet(num_classes, pretrain=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    since = time.time()
    best_acc = 0
    best = 0
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 10)


        # Iterate over data.
        running_loss = 0.0
        running_corrects = 0
        model.train()
        with torch.set_grad_enabled(True):
            for i, (inputs, labels) in enumerate(data_train):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data) * 1. / inputs.size(0)
                print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(data_train), loss.item()), end="")

                sys.stdout.flush()

        avg_loss = running_loss / len(data_train)
        t_acc = running_corrects.double() / len(data_train)

        running_loss = 0.0
        running_corrects = 0
        model.eval()
        with torch.set_grad_enabled(False):
            for i, (inputs, labels) in enumerate(data_val):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data) * 1. / inputs.size(0)

        val_loss = running_loss / len(data_val)
        val_acc = running_corrects.double() / len(data_val)

        print()
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print('lr rate: {:.6f}'.format(optimizer.param_groups[0]['lr']))
        print()

        if val_acc > best_acc:
            best_acc = val_acc
            best = epoch + 1

        lr_scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))

    return model

def test(model, data_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    model.eval()  # Set model to evaluate mode

    running_corrects = 0

    # Iterate over data.
    with torch.set_grad_enabled(False):
        for i, (inputs, labels) in enumerate(data_test):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data) * 1. / inputs.size(0)

    epoch_acc = running_corrects.double() / len(data_test)
    print()
    print('Test Acc: {:.4f}'.format(epoch_acc))
    print()

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    data_dir = '/data/dataset/tiny-imagenet-200'
    data_train, data_val, data_test = get_data_loader(data_dir)
    model = train(data_train, data_val, 200, 40, [20, 30])
    test(model, data_test)
