import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminator import build_discriminator
from .generator import buiid_generator

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def forward(self, x, real_img):
        bs, c, h, w = x.size()
        z = x.view(bs, -1)
        fake = self.generator(z)
        fake_img = fake.view(bs, c, h, w)
        dis_real = self.discriminator(real_img)
        dis_fake = self.discriminator(fake_img)
        out = {'fake_out': dis_fake, 'real_out': dis_real, 'fake_img': fake_img}
        
        return out
    
class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

        
    def gen_loss(self, output):

        G_fake = output['fake_out']        
        target = torch.ones_like(G_fake)
        loss_G = F.binary_cross_entropy(G_fake, target)
        losses = {"loss_G": loss_G}
        
        return losses
    
    def dis_loss(self, output):
        
        pred_fake = output['fake_out']
        pred_real = output['real_out']
        
        target_fake = torch.zeros_like(pred_fake)
        target_real = torch.ones_like(pred_real)        
        
        loss_fake = F.binary_cross_entropy(pred_fake, target_fake)
        loss_real = F.binary_cross_entropy(pred_real, target_real)
                
        loss_D = loss_real + loss_fake
        losses = {"loss_D": loss_D}
        
        return losses
    
    def forward(self, losses, output):
        gen_losses = self.gen_loss(output)
        losses.update(gen_losses)
        
        dis_losses = self.dis_loss(output)
        losses.update(dis_losses)
        
        return losses
        
        
def build_model(args):
    generator = buiid_generator(args)
    discriminator = build_discriminator(args)
    
    model = GAN(generator, discriminator)
    criterion = Criterion()
    
    return model, criterion