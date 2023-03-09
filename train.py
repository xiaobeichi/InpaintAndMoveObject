import os.path
import shutil
from configs.config import get_config
from utils.scheduler import MipLRDecay
from utils.loss import NeRFLoss, mse_to_psnr
from models.mipnerf import MipNeRF
import torch
import torch.optim as optim 
import torch.utils.tensorboard as tb
from os import path
from datasets.datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm
import cv2


def train_model(config):
    model_save_path = path.join(config.save_dir, "model.pt")
    optimizer_save_path = path.join(config.save_dir, "optim.pt")

    train_loader = get_dataloader(dataset_name=config.dataset_name, base_dir=config.data_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)
    data = iter(cycle(train_loader))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.data_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))
    test_data = get_dataloader(config.dataset_name, config.data_dir, split="test_image", factor=config.factor, shuffle=False)
        
    model = MipNeRF(
        train_loader.n_image,
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    if config.continue_training:
        model.load_state_dict(torch.load(model_save_path))
        optimizer.load_state_dict(torch.load(optimizer_save_path))

    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    print("n_train: ", train_loader.n_image)
    #u = torch.nn.parameter(torch.ones(train_loader.n_image, requires_grad=True))
    loss_func = NeRFLoss(model.u, config, device=config.device)
    model.train()

    os.makedirs(config.save_dir, exist_ok=True)
    shutil.rmtree(path.join(config.save_dir, 'train'), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(config.save_dir, 'train'), flush_secs=1)

    for step in tqdm(range(0, config.max_steps)):
        rays, pixels = next(data)
        comp_rgb, dist, acc = model(rays)
        if step % 1000 == 0: # show dist and acc
            print("dist: ",dist.shape, dist)
            print("acc: ",acc.shape, acc)
        pixels = pixels.to(config.device)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device), rays.indice, rays.isMask, step)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)
        
        if (step+1) % (config.save_every/5) == 0:
            print(step," train loss: ", float(loss_val.detach().cpu().numpy()), " psrn: ", float(np.mean(psnr)))

        if (step+1) % config.save_every == 0:
            if eval_data:
                del rays
                del pixels
                psnr = eval_model(config, model, eval_data)
                psnr = psnr.detach().cpu().numpy()
                logger.add_scalar('eval/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
                logger.add_scalar('eval/fine_psnr', float(psnr[-1]), global_step=step)
                logger.add_scalar('eval/avg_psnr', float(np.mean(psnr)), global_step=step)
                print(step, " eval psrn: ", float(np.mean(psnr)))
                
                torch.save(model.state_dict(), model_save_path)
                torch.save(optimizer.state_dict(), optimizer_save_path)

                # log test image
        if (step+1) % config.test_image == 0:
            model.eval()
            #rgb_frames = []
            i = 0
            print("test ", len(test_data), " images")
            for ray in tqdm(test_data):
                img, dist, acc = model.render_image(ray, test_data.h, test_data.w, chunks=config.chunks)
                # print("img:",img.shape)
                # print("imgmaz:", img.max())
                #rgb_frames.append(img)
                #Image.
                #cv2.imwrite(path.join(config.save_dir,str(step)+"_"+str(i)+".png"), img)
                logger.add_image("color"+str(i), img, step, dataformats="HWC")
                i += 1
                if i == 2:
                    break
            print("save image")
            model.train()
                



def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])


if __name__ == "__main__":
    config = get_config()
    train_model(config)
