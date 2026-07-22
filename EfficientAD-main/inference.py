import argparse
import json
import os

import cv2
import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common import get_autoencoder, ImageFolderWithoutTarget

DEFAULT_IMAGE_SIZE = 256
DEFAULT_OUT_CHANNELS = 384
on_gpu = torch.cuda.is_available()

default_transform = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_models(teacher_path, student_path, ae_path):
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    student = torch.load(student_path, map_location='cpu', weights_only=False)
    autoencoder = torch.load(ae_path, map_location='cpu', weights_only=False)

    teacher.eval()
    student.eval()
    autoencoder.eval()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    return teacher, student, autoencoder


def compute_norm_params(teacher, student, autoencoder, train_dir, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        teacher_mean = torch.tensor(data['teacher_mean']).view(1, -1, 1, 1)
        teacher_std = torch.tensor(data['teacher_std']).view(1, -1, 1, 1)
        q_st_start = torch.tensor(data['q_st_start'])
        q_st_end = torch.tensor(data['q_st_end'])
        q_ae_start = torch.tensor(data['q_ae_start'])
        q_ae_end = torch.tensor(data['q_ae_end'])
        if on_gpu:
            teacher_mean = teacher_mean.cuda()
            teacher_std = teacher_std.cuda()
            q_st_start = q_st_start.cuda()
            q_st_end = q_st_end.cuda()
            q_ae_start = q_ae_start.cuda()
            q_ae_end = q_ae_end.cuda()
        return teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end

    dataset = ImageFolderWithoutTarget(
        train_dir, transform=transforms.Lambda(lambda x: default_transform(x)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    teacher_mean, teacher_std = _teacher_normalization(teacher, loader)
    q_st_start, q_st_end, q_ae_start, q_ae_end = _map_normalization(
        teacher, student, autoencoder, loader, teacher_mean, teacher_std)

    if cache_path:
        save_data = {
            'teacher_mean': teacher_mean.cpu().flatten().tolist(),
            'teacher_std': teacher_std.cpu().flatten().tolist(),
            'q_st_start': q_st_start.cpu().item(),
            'q_st_end': q_st_end.cpu().item(),
            'q_ae_start': q_ae_start.cpu().item(),
            'q_ae_end': q_ae_end.cpu().item(),
        }
        with open(cache_path, 'w') as f:
            json.dump(save_data, f)

    return teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end


def run_inference(image_path, teacher, student, autoencoder,
                  teacher_mean, teacher_std,
                  q_st_start, q_st_end, q_ae_start, q_ae_end,
                  output_map_path=None, threshold=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    h_orig, w_orig = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(image_rgb)
    tensor = default_transform(pil_img).unsqueeze(0)
    if on_gpu:
        tensor = tensor.cuda()

    with torch.no_grad():
        map_combined, _, _ = _predict(
            tensor, teacher, student, autoencoder,
            teacher_mean, teacher_std,
            q_st_start, q_st_end, q_ae_start, q_ae_end)

    map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
    map_combined = torch.nn.functional.interpolate(
        map_combined, (h_orig, w_orig), mode='bilinear')
    anomaly_map = map_combined[0, 0].cpu().numpy()
    score = float(np.max(anomaly_map))

    if output_map_path:
        tifffile.imwrite(output_map_path, anomaly_map)

    result = {'anomaly_map': anomaly_map, 'score': score}
    if threshold is not None:
        result['is_anomaly'] = score > threshold
    return result


@torch.no_grad()
def _predict(image, teacher, student, autoencoder,
             teacher_mean, teacher_std,
             q_st_start, q_st_end, q_ae_start, q_ae_end):
    out_channels = DEFAULT_OUT_CHANNELS
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean(
        (teacher_output - student_output[:, :out_channels]) ** 2,
        dim=1, keepdim=True)
    map_ae = torch.mean(
        (autoencoder_output - student_output[:, out_channels:]) ** 2,
        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def _teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for img in tqdm(train_loader, desc='Computing teacher mean'):
        if on_gpu:
            img = img.cuda()
        output = teacher(img)
        mean_outputs.append(torch.mean(output, dim=[0, 2, 3]))
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for img in tqdm(train_loader, desc='Computing teacher std'):
        if on_gpu:
            img = img.cuda()
        output = teacher(img)
        distance = (output - channel_mean) ** 2
        mean_distances.append(torch.mean(distance, dim=[0, 2, 3]))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)
    return channel_mean, channel_std


@torch.no_grad()
def _map_normalization(teacher, student, autoencoder, loader, teacher_mean, teacher_std):
    out_channels = DEFAULT_OUT_CHANNELS
    maps_st = []
    maps_ae = []
    for img in tqdm(loader, desc='Computing map normalization'):
        if on_gpu:
            img = img.cuda()
        teacher_output = teacher(img)
        teacher_output = (teacher_output - teacher_mean) / teacher_std
        student_output = student(img)
        ae_output = autoencoder(img)
        map_st = torch.mean(
            (teacher_output - student_output[:, :out_channels]) ** 2,
            dim=1, keepdim=True)
        map_ae = torch.mean(
            (ae_output - student_output[:, out_channels:]) ** 2,
            dim=1, keepdim=True)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='inference')

    parser.add_argument('--teacher', default='output/1/trainings/mvtec_ad/my_product/teacher_final.pth')
    parser.add_argument('--student', default='output/1/trainings/mvtec_ad/my_product/student_final.pth')
    parser.add_argument('--autoencoder', default='output/1/trainings/mvtec_ad/my_product/autoencoder_final.pth')
    parser.add_argument('--train-dir', default='../mydataset/my_product/train',
                        help='Path to train/good/ for computing normalization')
    parser.add_argument('--norm-cache', default='output/1/trainings/mvtec_ad/my_product/norm_params.json',
                        help='Cache file for normalization params')
    parser.add_argument('--image', required=True, help='Input image to infer')
    parser.add_argument('--output-map', '-o', default=None, help='Output anomaly map .tiff path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Anomaly score threshold for classification')

    args = parser.parse_args()

    teacher, student, autoencoder = load_models(
        args.teacher, args.student, args.autoencoder)

    teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end = \
        compute_norm_params(teacher, student, autoencoder, args.train_dir, args.norm_cache)

    result = run_inference(
        args.image, teacher, student, autoencoder,
        teacher_mean, teacher_std,
        q_st_start, q_st_end, q_ae_start, q_ae_end,
        output_map_path=args.output_map, threshold=args.threshold)

    print(f"Anomaly score: {result['score']:.6f}")
    if 'is_anomaly' in result:
        print(f"Result: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")
