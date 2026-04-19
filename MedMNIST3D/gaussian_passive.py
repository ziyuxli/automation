import argparse
import math
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from medmnist import INFO
from models import ResNet18, ResNet50
from utils import Transform3D, model_to_syncbn


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedSubset(data.Dataset):
    """
    Same as torch.utils.data.Subset, but also returns original sample index.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        original_idx = self.indices[i]
        x, y = self.dataset[original_idx]
        return x, y, original_idx


class GaussianDropoutHookManager:
    """
    Add multiplicative Gaussian noise to Conv/Linear outputs during train() mode:
        output <- output * N(1, std^2)
    where std^2 = p / (1-p)

    This lets the same model support MC sampling without rewriting model definitions.
    """
    def __init__(self, model, drop_prob=0.2, apply_to_linear=True, apply_to_conv=True):
        assert 0.0 <= drop_prob < 1.0
        self.model = model
        self.drop_prob = drop_prob
        keep_prob = 1.0 - drop_prob
        self.std = math.sqrt(drop_prob / keep_prob) if drop_prob > 0 else 0.0
        self.handles = []

        module_types = []
        if apply_to_conv:
            module_types.extend([nn.Conv2d, nn.Conv3d])
        if apply_to_linear:
            module_types.extend([nn.Linear])
        module_types = tuple(module_types)

        for module in self.model.modules():
            if isinstance(module, module_types):
                h = module.register_forward_hook(self._hook_fn)
                self.handles.append(h)

    def _hook_fn(self, module, inputs, output):
        if not module.training:
            return output
        if self.drop_prob <= 0:
            return output
        if not torch.is_tensor(output):
            return output

        noise = torch.randn_like(output) * self.std + 1.0
        return output * noise

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def set_mc_sampling_mode(model):
    """
    Turn on stochastic Gaussian-dropout forward passes,
    but freeze BN / SyncBN to avoid updating running stats.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def unpack_batch(batch):
    """
    Support both:
      - normal dataset: (x, y)
      - indexed subset: (x, y, idx)
    """
    if len(batch) == 2:
        x, y = batch
        idx = None
    elif len(batch) == 3:
        x, y, idx = batch
    else:
        raise ValueError(f'Unexpected batch size tuple length: {len(batch)}')
    return x, y, idx


def build_model(model_flag, n_channels, n_classes, conv, pretrained_3d):
    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    if conv == 'ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    elif conv == 'Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    elif conv == 'Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
    else:
        raise ValueError(f'Unknown conv type: {conv}')

    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []

    for batch in train_loader:
        inputs, targets, _ = unpack_batch(batch)

        inputs = inputs.to(device)
        targets = torch.squeeze(targets, 1).long().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, evaluator, data_loader, criterion, device):
    model.eval()
    total_loss = []
    y_score = []

    for batch in data_loader:
        inputs, targets, _ = unpack_batch(batch)

        inputs = inputs.to(device)
        targets = torch.squeeze(targets, 1).long().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        probs = torch.softmax(outputs, dim=1)

        total_loss.append(loss.item())
        y_score.append(probs.detach().cpu())

    y_score = torch.cat(y_score, dim=0).numpy()
    auc, acc = evaluator.evaluate(y_score)
    test_loss = float(np.mean(total_loss)) if total_loss else 0.0

    return [test_loss, auc, acc]


@torch.no_grad()
def mc_predict_probs(model, loader, device, mc_samples):
    """
    Multiple stochastic forward passes for active learning acquisition.
    Returns:
        mean_probs: [N, C]
        all_probs:  [T, N, C]
        all_indices: [N]
    """
    all_mc_probs = []
    all_indices = None

    for _ in range(mc_samples):
        set_mc_sampling_mode(model)

        probs_this_pass = []
        indices_this_pass = []

        for batch in loader:
            inputs, _, indices = unpack_batch(batch)

            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).detach().cpu()

            probs_this_pass.append(probs)
            indices_this_pass.extend(indices.tolist())

        probs_this_pass = torch.cat(probs_this_pass, dim=0)
        all_mc_probs.append(probs_this_pass)

        if all_indices is None:
            all_indices = indices_this_pass

    all_probs = torch.stack(all_mc_probs, dim=0)   # [T, N, C]
    mean_probs = all_probs.mean(dim=0)             # [N, C]
    return mean_probs, all_probs, all_indices


def predictive_entropy(mean_probs, eps=1e-8):
    return -(mean_probs * mean_probs.clamp_min(eps).log()).sum(dim=1)


def expected_entropy(all_probs, eps=1e-8):
    ent = -(all_probs * all_probs.clamp_min(eps).log()).sum(dim=2)  # [T, N]
    return ent.mean(dim=0)


def bald_score(mean_probs, all_probs, eps=1e-8):
    return predictive_entropy(mean_probs, eps) - expected_entropy(all_probs, eps)


def select_samples(
    strategy,
    model,
    full_train_dataset,
    unlabeled_indices,
    batch_size,
    device,
    mc_samples,
    samples_per_round,
):
    """
    strategy:
      - passive / random : random selection
      - entropy          : MC Gaussian dropout + predictive entropy
      - bald             : MC Gaussian dropout + BALD
    """
    n_select = min(samples_per_round, len(unlabeled_indices))

    if strategy in ['passive', 'random']:
        selected = np.random.choice(unlabeled_indices, size=n_select, replace=False).tolist()
        scores = None
        return selected, scores

    unlabeled_dataset = IndexedSubset(full_train_dataset, unlabeled_indices)
    unlabeled_loader = data.DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    mean_probs, all_probs, pool_indices = mc_predict_probs(
        model=model,
        loader=unlabeled_loader,
        device=device,
        mc_samples=mc_samples,
    )

    if strategy == 'entropy':
        scores = predictive_entropy(mean_probs)
    elif strategy == 'bald':
        scores = bald_score(mean_probs, all_probs)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    scores_np = scores.numpy()
    selected_order = np.argsort(-scores_np)[:n_select]
    selected = [pool_indices[i] for i in selected_order]
    selected_scores = [float(scores_np[i]) for i in selected_order]
    return selected, selected_scores


def main(data_flag, output_root, samples_per_round, max_epochs,
         gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag,
         as_rgb, shape_transform, run, initial_size=200,
         strategy='passive', gaussian_drop_prob=0.2, mc_samples=20,
         seed=42):

    seed_everything(seed)
    lr = 0.001

    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        idx = int(str_id)
        if idx >= 0:
            gpu_ids.append(idx)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    device = torch.device('cuda:0') if gpu_ids else torch.device('cpu')

    output_root = os.path.join(
        output_root,
        data_flag,
        model_flag + "_" + run,
        time.strftime("%y%m%d_%H%M%S")
    )
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
    model = build_model(model_flag, n_channels, n_classes, conv, pretrained_3d)
    model = model.to(device)

    # Gaussian dropout hook is only really used when strategy is entropy/bald,
    # but attaching it here keeps the script unified.
    gd_manager = GaussianDropoutHookManager(
        model=model,
        drop_prob=gaussian_drop_prob,
        apply_to_linear=True,
        apply_to_conv=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_indices = list(range(len(full_train_dataset)))
    np.random.shuffle(all_indices)
    labeled_indices = all_indices[:initial_size]
    unlabeled_indices = all_indices[initial_size:]

    log_path = os.path.join(output_root, f'{data_flag}_{strategy}_log.txt')
    plot_dir = os.path.join(output_root, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    labeled_sizes = []
    test_losses, test_aucs, test_accs = [], [], []

    print(f'==> Starting {strategy} learning...')
    num_rounds = len(unlabeled_indices) // samples_per_round
    if len(unlabeled_indices) % samples_per_round != 0:
        num_rounds += 1

    print(f'Train set size: {len(all_indices)}, initial labeled: {initial_size}, '
          f'unlabeled: {len(unlabeled_indices)}, rounds to run: {num_rounds}')

    def run_round(round_label, loader, labeled_count):
        prev_val_loss = float('inf')
        final_val_metrics = None
        final_test_metrics = None

        with open(log_path, 'a') as f:
            f.write(f'{round_label} labeled={labeled_count}\n')

        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_evaluator, val_loader, criterion, device)

            epoch_log = (
                f'  epoch {epoch + 1}  train loss: {train_loss:.5f}  '
                f'val loss: {val_metrics[0]:.5f}  auc: {val_metrics[1]:.5f}  acc: {val_metrics[2]:.5f}'
            )
            print(epoch_log)

            with open(log_path, 'a') as f:
                f.write(epoch_log + '\n')

            final_val_metrics = val_metrics
            if val_metrics[0] >= prev_val_loss:
                print(f'  Early stop at epoch {epoch + 1} (val loss did not decrease)')
                break
            prev_val_loss = val_metrics[0]

        final_test_metrics = evaluate(model, test_evaluator, test_loader, criterion, device)
        summary = ('  test  loss: %.5f  auc: %.5f  acc: %.5f\n'
                   % (final_test_metrics[0], final_test_metrics[1], final_test_metrics[2]))
        print(summary)

        with open(log_path, 'a') as f:
            f.write(summary)

        return final_val_metrics, final_test_metrics

    # Round 0
    print(f'\n[Round 0] Initial training on {initial_size} samples...')
    initial_loader = data.DataLoader(
        IndexedSubset(full_train_dataset, labeled_indices),
        batch_size=batch_size,
        shuffle=True
    )
    _, test_metrics = run_round('[Round 0]', initial_loader, initial_size)
    labeled_sizes.append(initial_size)
    test_losses.append(test_metrics[0])
    test_aucs.append(test_metrics[1])
    test_accs.append(test_metrics[2])

    for round_idx in range(num_rounds):
        if len(unlabeled_indices) == 0:
            break

        selected, selected_scores = select_samples(
            strategy=strategy,
            model=model,
            full_train_dataset=full_train_dataset,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_size,
            device=device,
            mc_samples=mc_samples,
            samples_per_round=samples_per_round,
        )

        labeled_indices.extend(selected)
        selected_set = set(selected)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_set]

        print(f'\n[Round {round_idx + 1}/{num_rounds}] Strategy: {strategy}, '
              f'Labeled: {len(labeled_indices)}, Added: {len(selected)}, '
              f'Unlabeled remaining: {len(unlabeled_indices)}')

        if selected_scores is not None:
            print(f'  Acquisition score: min={min(selected_scores):.6f}, '
                  f'max={max(selected_scores):.6f}, mean={np.mean(selected_scores):.6f}')

        train_loader = data.DataLoader(
            IndexedSubset(full_train_dataset, labeled_indices),
            batch_size=batch_size,
            shuffle=True
        )

        _, test_metrics = run_round(f'[Round {round_idx + 1}]', train_loader, len(labeled_indices))
        labeled_sizes.append(len(labeled_indices))
        test_losses.append(test_metrics[0])
        test_aucs.append(test_metrics[1])
        test_accs.append(test_metrics[2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, values, ylabel in zip(
        axes,
        [test_losses, test_aucs, test_accs],
        ['Loss', 'AUC', 'ACC']
    ):
        ax.plot(labeled_sizes, values, marker='o', markersize=3, linewidth=1)
        ax.set_xlabel('Labeled set size')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Test {ylabel} vs Labeled Size')
        ax.grid(True)

    fig.tight_layout()
    plot_path = os.path.join(plot_dir, f'test_metrics_per_round_{strategy}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    gd_manager.remove()

    print(f'Plot saved to {plot_path}')
    print(f'Done. Log saved to {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Passive / Active Learning for MedMNIST3D with optional Gaussian Dropout')

    parser.add_argument('--data_flag', default='organmnist3d', type=str)
    parser.add_argument('--output_root', default='./output', type=str)
    parser.add_argument('--samples_per_round', default=10, type=int,
                        help='number of selected samples added per round')
    parser.add_argument('--max_epochs', default=5, type=int,
                        help='max epochs per round; stops early if val loss does not decrease')
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
    parser.add_argument('--initial_size', default=200, type=int,
                        help='number of labeled samples to start with')

    # New arguments
    parser.add_argument('--strategy', default='passive', type=str,
                        choices=['passive', 'random', 'entropy', 'bald'],
                        help='sample selection strategy')
    parser.add_argument('--gaussian_drop_prob', default=0.2, type=float,
                        help='Gaussian dropout probability for MC sampling')
    parser.add_argument('--mc_samples', default=20, type=int,
                        help='number of stochastic forward passes for active learning')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    main(
        data_flag=args.data_flag,
        output_root=args.output_root,
        samples_per_round=args.samples_per_round,
        max_epochs=args.max_epochs,
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
        initial_size=args.initial_size,
        strategy=args.strategy,
        gaussian_drop_prob=args.gaussian_drop_prob,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )