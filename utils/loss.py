import torch


class NeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self, u, config, device="cuda"):
        super(NeRFLoss, self).__init__()
        self.coarse_weight_decay = config.coarse_weight_decay
        self.reg_weight = config.reg_weight
        self.device = device
        # print("n_train: ",n_train)
        self.u = u
        # self.u = torch.nn.parameter(torch.ones(n_train), requires_grad=True)

    def forward(self, input, target, mask, indice, isMask, step):
        #print("indiceshape : ",indice.shape," ",indice[0][0])
        rgb_losses = []
        #u2 = torch.nn.ReLU()(self.u)
        u2 = torch.clamp(self.u, 0, 5)
        reg_loss = torch.sum(u2)/len(self.u)
        psnrs = []
        for rgb in input:
            mse = 0
            for i in range(len(rgb)):
                if isMask[i]:
                    # wu!!!!
                    #print("mask: ", (mask[i] * ((rgb[i] - target[i, :3]) ** 2)).sum() * torch.exp(-u2[indice[i]]))
                    mse = mse + (mask[i] * ((rgb[i] - target[i, :3]) ** 2)).sum() * torch.exp(-u2[indice[i][0].long()])
                else:
                    mse = mse + (mask[i] * ((rgb[i] - target[i, :3]) ** 2)).sum()
            mse = mse / mask.sum()
            mse_psrn = (mask * ((rgb - target[..., :3]) ** 2)).sum() / mask.sum()
            
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse_psrn))
            rgb_losses.append(mse)
        losses = torch.stack(rgb_losses)
        loss = self.coarse_weight_decay * torch.sum(losses[:-1]) + losses[-1] + self.reg_weight * reg_loss
        if(step % 1000 == 0):
            print("U: ", self.u)
            print("rgbloss: ", self.coarse_weight_decay * torch.sum(losses[:-1]) + losses[-1])
            print("regloss: ", self.reg_weight * reg_loss)
        # distortion_loss?
        return loss, torch.Tensor(psnrs)


def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)
