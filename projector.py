import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from landmarks import FaceAlignment
from landmarks import LandmarksType

import landmark_utils
from landmark_utils import *

import lpips
from model import Generator
import id_loss


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument("--landmarks", type=float, default=0, help="weight of landmark loss")
    parser.add_argument(
        "--landmark_augmentation",
        type=str,
        default=None,
        choices=[None, 'smile', 'raise_left_eyebrow', 'raise_right_eyebrow', 'raise_eyebrows'],
        help="type of augmentation we want to perform with landmarks"
    )
    parser.add_argument(
        "--landmark_scale",
        type=float,
        default=1,
        help="scale for landmark augmentation"
    )
    parser.add_argument("--plot_target_landmarks", action="store_true", help="whether to plot and save target landmarks")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument("--identity", type=float, default=0.2, help="weight of identity loss")
    parser.add_argument("--perceptual", type=float, default=1.0, help="weight of perceptual loss")
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []
    if args.landmarks > 0:
        bbox = [[0,0,255,255]]
        fa = FaceAlignment(LandmarksType._2D, device='cuda')
        target_landmarks = torch.zeros((len(args.files), 68, 2))
        for i, imgfile in enumerate(args.files):
            landmarks, heatmaps = fa.get_landmarks(imgfile, bbox)
            if args.landmark_augmentation is not None:
                transformation = getattr(landmark_utils, args.landmark_augmentation)
                landmarks = transformation(landmarks, 256, args.landmark_scale)
                hm_transformation = getattr(landmark_utils, args.landmark_augmentation + '_hm')
                target_heatmaps = hm_transformation(heatmaps, args.landmark_scale)*100
            if args.plot_target_landmarks is not None:
                my_landmarks = np.zeros((68, 2))
                for j, lm in enumerate(target_heatmaps):
                    coords = np.unravel_index(np.argmax(lm.detach().cpu().numpy()), (64, 64))
                    my_landmarks[j, 0] = coords[1]
                    my_landmarks[j, 1] = coords[0]
                my_landmarks *= 4
                landmarks = my_landmarks
                arr = np.zeros((256, 256))
                plt.imshow(Image.open(imgfile))
                plt.scatter(landmarks[:, 0], landmarks[:, 1], 3, c='greenyellow')
                plt.tight_layout()
                plt.savefig('results/landmarks.png')
            print(target_landmarks[i, :, :].shape, landmarks.shape)
            target_landmarks[i, :, :] = torch.from_numpy(landmarks)

    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    identity = id_loss.IDLoss().to('cuda').eval()

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []
    losses = np.zeros((4, len(pbar)))

    for j in pbar:
        t = j / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])
        if args.landmarks > 0:
            fa_input = img_gen.clamp_(min=-1, max=1).add(1).div_(2).mul(255).permute(0, 2, 3, 1).requires_grad_(True)
            lm, hm = fa.get_landmarks(fa_input, bbox)
            lm_loss = F.mse_loss(hm*100, target_heatmaps)
        else:
            lm_loss = torch.tensor(0)

        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)
        identity_loss, _, _ = identity(img_gen.cuda(), img.unsqueeze(0).cuda(), img.unsqueeze(0).cuda())

        losses[0, j] = p_loss * args.perceptual
        losses[1, j] = mse_loss * args.mse
        losses[2, j] = identity_loss * args.identity
        losses[3, j] = lm_loss * args.landmarks

        loss = args.perceptual * p_loss + args.noise_regularize * n_loss + args.mse * mse_loss + \
               lm_loss * args.landmarks + identity_loss * args.identity

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        noise_normalize_(noises)

        if (j + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; landmarks: {lm_loss.item():.4f}; identity: {identity_loss.item():.4f}"
            )
        )

    x = np.arange(len(pbar))
    plt.figure(0)
    plt.plot(x, losses[0, :], label="Perceptual loss")
    plt.plot(x, losses[1, :], label="L2 loss")
    plt.plot(x, losses[2, :], label="Identity loss")
    plt.plot(x, losses[3, :], label="Landmark loss")
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("Progress of losses during optimization")
    plt.legend()
    plt.savefig('results/losses.png')

    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

    img_ar = make_image(img_gen)

    result_file = {}
    for j, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[j: j + 1])

        result_file[input_name] = {
            "img": img_gen[j],
            "latent": latent_in[j],
            "noise": noise_single,
        }

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_T%H-%M-%S")
        img_name = "results/" + os.path.splitext(os.path.basename(input_name))[0] + dt_string + ".png"
        pil_img = Image.fromarray(img_ar[j])
        pil_img.save(img_name)

    torch.save(result_file, filename)
