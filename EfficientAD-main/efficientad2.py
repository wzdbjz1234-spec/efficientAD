#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from functools import partial
import itertools
import json
import os
import random
import shutil
from tqdm import tqdm
from common import get_autoencoder, get_autoencoder_tiny, get_pdn_small, get_pdn_medium, get_pdn_tiny, ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium', 'tiny'])
    parser.add_argument('-w', '--weights', default='')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('--ae-channels', type=int, default=384,
                        help='Autoencoder output channels (default 384). '
                             'Lower values reduce Student+AE parameters. '
                             'Teacher always uses 384 channels.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument(
        '--amp', choices=['auto', 'off', 'bf16'], default='auto',
        help='Mixed precision mode. "auto" enables BF16 on supported CUDA GPUs.')
    parser.add_argument(
        '--hard-mining-ratio', type=float, default=0.001,
        help='Fraction of largest student/teacher errors used by hard mining.')
    parser.add_argument(
        '--mask-config', default='auto',
        help='ROI JSON containing masks. "auto" checks the product directory, '
             'repository roi_config.json, and templates/<subdataset>/roi.json; '
             'use "none" to disable masking.')
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
teacher_channels = 384  # Teacher output channels (fixed, pre-trained)
image_size = 256
# ae_channels is set from CLI (default 384) — controls Autoencoder output
# channels.  Student outputs teacher_channels + ae_channels.

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.Compose([
    transforms.ColorJitter(brightness=(0, 2.0)),
    transforms.ColorJitter(saturation=(0, 2.5)),
    transforms.ColorJitter(hue=(-0.3, 0.3))
])

def train_transform(image, valid_input_mask=None):
    image_st = default_transform(image)
    image_ae = default_transform(transform_ae(image))
    if valid_input_mask is not None:
        image_st = image_st * valid_input_mask[0]
        image_ae = image_ae * valid_input_mask[0]
    return image_st, image_ae


def load_training_mask(mask_config, dataset_path, subdataset):
    if mask_config.lower() == 'none':
        return None, None

    if mask_config != 'auto':
        candidates = [os.path.abspath(mask_config)]
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(dataset_path, subdataset, 'roi_config.json'),
            os.path.join(project_root, 'templates', subdataset, 'roi.json'),
            os.path.join(project_root, 'roi_config.json'),
        ]

    existing = [path for path in candidates if os.path.isfile(path)]
    if not existing:
        if mask_config != 'auto':
            raise FileNotFoundError(f'Mask config not found: {candidates[0]}')
        print('Training mask: none (no ROI mask config found)')
        return None, None

    selected = None
    data = None
    raw_masks = None
    for candidate in existing:
        with open(candidate, encoding='utf-8') as handle:
            candidate_data = json.load(handle)
        candidate_masks = candidate_data.get('masks')
        if candidate_masks is None:
            candidate_mask = candidate_data.get('mask')
            candidate_masks = (
                [] if candidate_mask is None else [candidate_mask])
        if candidate_masks or mask_config != 'auto':
            selected = candidate
            data = candidate_data
            raw_masks = candidate_masks
            break

    if selected is None:
        print('Training mask: none (ROI configs contain no masks)')
        return None, existing[0]

    roi = tuple(map(int, data['roi']))
    masks = [tuple(map(int, mask)) for mask in raw_masks]
    if not masks:
        print(f'Training mask: none ({selected} contains no masks)')
        return None, selected

    _, _, roi_width, roi_height = roi
    valid = torch.ones((1, 1, roi_height, roi_width), dtype=torch.float32)
    for x, y, width, height in masks:
        if (width <= 0 or height <= 0 or x < 0 or y < 0
                or x + width > roi_width or y + height > roi_height):
            raise ValueError(
                f'Mask ({x},{y},{width},{height}) is outside '
                f'ROI {roi_width}x{roi_height}')
        valid[:, :, y:y + height, x:x + width] = 0
    valid = F.interpolate(
        valid, size=(image_size, image_size), mode='nearest')
    if not torch.any(valid):
        raise ValueError('Training mask excludes the entire ROI')
    print(f'Training mask: {len(masks)} region(s) loaded from {selected}')
    return valid, selected


def feature_valid_mask(valid_input_mask, reference):
    if valid_input_mask is None:
        return None
    valid_input_mask = valid_input_mask.to(reference.device)
    return F.interpolate(
        valid_input_mask, size=reference.shape[-2:], mode='nearest').bool()


def masked_mean(value, valid_input_mask):
    mask = feature_valid_mask(valid_input_mask, value)
    if mask is None:
        return torch.mean(value)
    return torch.mean(value.masked_select(mask.expand_as(value)))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()
    if config.batch_size <= 0:
        raise ValueError('--batch-size must be greater than zero')
    if config.num_workers < 0:
        raise ValueError('--num-workers must be non-negative')
    if config.prefetch_factor <= 0:
        raise ValueError('--prefetch-factor must be greater than zero')
    if not 0 < config.hard_mining_ratio <= 1:
        raise ValueError('--hard-mining-ratio must be in the interval (0, 1]')
    if config.ae_channels < 1:
        raise ValueError('--ae-channels must be greater than zero')

    ae_channels = config.ae_channels
    student_channels = teacher_channels + ae_channels
    print(f'Teacher channels: {teacher_channels}  '
          f'AE channels: {ae_channels}  '
          f'Student channels: {student_channels}')

    bf16_supported = on_gpu and torch.cuda.is_bf16_supported()
    if config.amp == 'bf16' and not bf16_supported:
        raise RuntimeError('BF16 AMP was requested but is not supported')
    amp_enabled = bf16_supported and config.amp != 'off'
    if on_gpu:
        torch.backends.cudnn.benchmark = True
    print('Training precision: {}'.format(
        'BF16 AMP' if amp_enabled else 'FP32'))

    if not config.weights:
        if config.model_size == 'tiny':
            config.weights = 'models/teacher_small.pth'
        else:
            config.weights = f'models/teacher_{config.model_size}.pth'

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    valid_input_mask, mask_config_path = load_training_mask(
        config.mask_config, dataset_path, config.subdataset)

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(
            partial(train_transform, valid_input_mask=valid_input_mask)))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(
                partial(train_transform, valid_input_mask=valid_input_mask)))
    else:
        raise Exception('Unknown config.dataset')


    loader_kwargs = {
        'num_workers': config.num_workers,
        'pin_memory': on_gpu,
    }
    if config.num_workers > 0:
        loader_kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': config.prefetch_factor,
        })
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        **loader_kwargs)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(
            penalty_set, batch_size=config.batch_size, shuffle=True,
            **loader_kwargs)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(teacher_channels)
        student = get_pdn_small(student_channels)
        autoencoder = get_autoencoder(ae_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(teacher_channels)
        student = get_pdn_medium(student_channels)
        autoencoder = get_autoencoder(ae_channels)
    elif config.model_size == 'tiny':
        teacher = get_pdn_small(teacher_channels)
        student = get_pdn_tiny(student_channels)
        autoencoder = get_autoencoder_tiny(ae_channels)
    else:
        raise Exception()
    device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
    )

    state_dict = torch.load(
        config.weights,
        map_location=device
    )

    # teacher frozen
    teacher.load_state_dict(state_dict)
    teacher.requires_grad_(False)
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(
        teacher, train_loader, valid_input_mask)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)

    log_interval = max(1, config.train_steps // 1000)
    ema_alpha = 0.05
    ema_losses = None

    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda(non_blocking=True)
            image_ae = image_ae.cuda(non_blocking=True)
            if image_penalty is not None:
                image_penalty = image_penalty.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16,
                enabled=amp_enabled):
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (
                    teacher_output_st - teacher_mean) / (teacher_std + 1e-6)
            student_output_st = student(image_st)[:, :teacher_channels]
            distance_st = (
                teacher_output_st.float() - student_output_st.float()) ** 2
            distance_st_valid = distance_st
            train_feature_mask = feature_valid_mask(
                valid_input_mask, distance_st)
            if train_feature_mask is not None:
                distance_st_valid = distance_st.masked_select(
                    train_feature_mask.expand_as(distance_st))

            hard_count = max(
                1, int(distance_st_valid.numel()
                       * config.hard_mining_ratio))
            loss_hard = torch.topk(
                distance_st_valid.flatten(), k=hard_count,
                sorted=False).values.mean()

            if image_penalty is not None:
                student_output_penalty = student(
                    image_penalty)[:, :teacher_channels]
                loss_penalty = torch.mean(student_output_penalty.float()**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard

            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (
                    teacher_output_ae - teacher_mean) / (teacher_std + 1e-6)
            student_output_ae = student(image_ae)[:, teacher_channels:]
            distance_ae = (
                teacher_output_ae.float() - ae_output.float()) ** 2
            distance_stae = (
                ae_output.float() - student_output_ae.float()) ** 2
            loss_ae = masked_mean(distance_ae, valid_input_mask)
            loss_stae = masked_mean(distance_stae, valid_input_mask)
            loss_total = loss_st + loss_ae + 2 * loss_stae

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        current_losses = torch.stack((
            loss_st.detach(), loss_ae.detach(), loss_stae.detach(),
            loss_total.detach()))
        with torch.no_grad():
            if ema_losses is None:
                ema_losses = current_losses
            else:
                ema_losses.lerp_(current_losses, ema_alpha)

        if iteration % log_interval == 0:
            current_total, ema_st, ema_ae, ema_stae, ema_total = torch.cat((
                loss_total.detach().reshape(1), ema_losses)).cpu().tolist()
            tqdm_obj.set_description(
                'L {:.4f}'.format(current_total))
            tqdm_obj.set_postfix(
                EMA='{:.4f}'.format(ema_total),
                ST='{:.4f}'.format(ema_st),
                AE='{:.4f}'.format(ema_ae),
                SAE='{:.4f}'.format(ema_stae),
                lr='{:.1e}'.format(scheduler.get_last_lr()[0]))

        if iteration > 0 and iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                             'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                             'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                 'autoencoder_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            tqdm_obj.clear()
            print('-' * 60)
            print(f'[Step {iteration}/{config.train_steps}] Intermediate evaluation...')
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                valid_input_mask=valid_input_mask,
                desc='Intermediate map normalization')
            auc = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                valid_input_mask=valid_input_mask,
                test_output_dir=None, desc='Intermediate inference')
            print('Intermediate image auc: {:.4f}'.format(auc))
            print('-' * 60)

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))
    if mask_config_path is not None and valid_input_mask is not None:
        saved_mask_config = os.path.join(train_output_dir, 'mask_config.json')
        if os.path.abspath(mask_config_path) != os.path.abspath(saved_mask_config):
            shutil.copyfile(mask_config_path, saved_mask_config)

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, valid_input_mask=valid_input_mask,
        desc='Final map normalization')
    print('\n' + '=' * 60)
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        valid_input_mask=valid_input_mask,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))
    print('=' * 60)

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference', valid_input_mask=None):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        if valid_input_mask is not None:
            image = image * valid_input_mask[0]
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            valid_input_mask=valid_input_mask)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        if valid_input_mask is not None:
            output_valid = F.interpolate(
                valid_input_mask, (orig_height, orig_width), mode='nearest')
            map_combined *= output_valid[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None,
            valid_input_mask=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :teacher_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, teacher_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start + 1e-6)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start + 1e-6)
    output_mask = feature_valid_mask(valid_input_mask, map_st)
    if output_mask is not None:
        map_st = map_st * output_mask
        map_ae = map_ae * output_mask
    map_combined = 0.2 * map_st + 0.8 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization',
                      valid_input_mask=None):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, valid_input_mask=valid_input_mask)
        output_mask = feature_valid_mask(valid_input_mask, map_st)
        if output_mask is None:
            maps_st.append(map_st.flatten())
            maps_ae.append(map_ae.flatten())
        else:
            maps_st.append(map_st.masked_select(output_mask))
            maps_ae.append(map_ae.masked_select(output_mask))
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader, valid_input_mask=None):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        output_mask = feature_valid_mask(valid_input_mask, teacher_output)
        if output_mask is None:
            mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        else:
            count = output_mask.sum() * teacher_output.shape[0]
            mean_output = (
                teacher_output * output_mask).sum(dim=[0, 2, 3]) / count
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        output_mask = feature_valid_mask(valid_input_mask, distance)
        if output_mask is None:
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
        else:
            count = output_mask.sum() * distance.shape[0]
            mean_distance = (
                distance * output_mask).sum(dim=[0, 2, 3]) / count
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
