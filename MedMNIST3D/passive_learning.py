import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50
from utils import Transform3D, model_to_syncbn


def main(data_flag, output_root, samples_per_round, epochs_per_round,
         gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag,
         as_rgb, shape_transform, run):

    lr = 0.001

    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

    output_root = os.path.join(output_root, data_flag, model_flag + run, time.strftime("%y%m%d_%H%M%S"))
    os.makedirs(output_root, exist_ok=True)

    print('==> Preparing data...')

    train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()

    full_train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb, size=size)

    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)

    criterion = nn.CrossEntropyLoss()

    print('==> Building model...')

    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    if conv == 'ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    if conv == 'Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if conv == 'Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pool of all training indices, start with empty labeled set
    all_indices = list(range(len(full_train_dataset)))
    np.random.shuffle(all_indices)
    labeled_indices = []
    unlabeled_indices = all_indices.copy()

    log_path = os.path.join(output_root, f'{data_flag}_passive_log.txt')

    print('==> Starting passive learning...')

    num_rounds = len(all_indices) // samples_per_round
    print(f'Train set size: {len(all_indices)}, rounds to run: {num_rounds}')

    for round_idx in range(num_rounds):
        # Randomly select samples_per_round from unlabeled pool
        n_select = min(samples_per_round, len(unlabeled_indices))
        selected = np.random.choice(unlabeled_indices, size=n_select, replace=False).tolist()
        labeled_indices.extend(selected)
        for s in selected:
            unlabeled_indices.remove(s)

        print(f'\n[Round {round_idx + 1}/{num_rounds}] Labeled pool size: {len(labeled_indices)}, '
              f'Added: {n_select}, Unlabeled remaining: {len(unlabeled_indices)}')

        # Build loader from current labeled set
        labeled_subset = data.Subset(full_train_dataset, labeled_indices)
        train_loader = data.DataLoader(dataset=labeled_subset, batch_size=batch_size, shuffle=True)

        # Train for epochs_per_round epochs (no model reinit)
        for epoch in range(epochs_per_round):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on val and test
        val_metrics = evaluate(model, val_evaluator, val_loader, criterion, device)
        test_metrics = evaluate(model, test_evaluator, test_loader, criterion, device)

        val_log = 'val   loss: %.5f  auc: %.5f  acc: %.5f' % (val_metrics[0], val_metrics[1], val_metrics[2])
        test_log = 'test  loss: %.5f  auc: %.5f  acc: %.5f' % (test_metrics[0], test_metrics[1], test_metrics[2])
        log_line = (f'[Round {round_idx + 1}] labeled={len(labeled_indices)}\n'
                    f'  {val_log}\n'
                    f'  {test_log}\n')
        print(log_line)

        with open(log_path, 'a') as f:
            f.write(log_line)

    print(f'Done. Log saved to {log_path}')


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def evaluate(model, evaluator, data_loader, criterion, device):
    model.eval()
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

    y_score = y_score.detach().cpu().numpy()
    auc, acc = evaluator.evaluate(y_score)
    test_loss = sum(total_loss) / len(total_loss)
    return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Passive Learning for MedMNIST3D')

    parser.add_argument('--data_flag', default='organmnist3d', type=str)
    parser.add_argument('--output_root', default='./output', type=str)
    parser.add_argument('--samples_per_round', default=10, type=int,
                        help='number of randomly selected samples added per round')
    parser.add_argument('--epochs_per_round', default=5, type=int,
                        help='number of training epochs per round')
    parser.add_argument('--size', default=28, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--conv', default='ACSConv',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv', type=str)
    parser.add_argument('--pretrained_3d', default='i3d', type=str)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--as_rgb', action='store_true')
    parser.add_argument('--shape_transform', action='store_true')
    parser.add_argument('--model_flag', default='resnet18',
                        help='choose backbone, resnet18/resnet50', type=str)
    parser.add_argument('--run', default='model1', type=str)

    args = parser.parse_args()

    main(
        data_flag=args.data_flag,
        output_root=args.output_root,
        samples_per_round=args.samples_per_round,
        epochs_per_round=args.epochs_per_round,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size,
        size=args.size,
        conv=args.conv,
        pretrained_3d=args.pretrained_3d,
        download=args.download,
        model_flag=args.model_flag,
        as_rgb=args.as_rgb,
        shape_transform=args.shape_transform,
        run=args.run,
    )
