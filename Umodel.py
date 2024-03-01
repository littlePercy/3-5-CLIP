import torch.nn as nn
import torch
import clip


class Unet(nn.Module):

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
        )
        return block

    def __init__(self, in_channel, out_channel, check_path=''):
        super(Unet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(256, 512)
        self.conv_maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode5 = self.contracting_block(512, 1024)
        # Bottleneck
        self.bottleneck = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                             output_padding=1)
        # Decode
        self.conv_decode1 = self.expansive_block(1024, 512, 256)
        self.conv_decode2 = self.expansive_block(512, 256, 128)
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

        # text Encoder
        device = torch.device("cuda" if torch.cuda.is_available() else cpu)
        self.clip_model, _ = clip.load('ViT-B/16', device=device, jit=False, download_root='checkpoints')
        # ========== 导入预训练模型 冻结预训练模型参数==========
        # checkpoint = torch.load(check_path)
        # self.clip_model.load_state_dict(checkpoint['network'])
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.Linear1 = nn.Linear(512, 64)
        self.Linear2 = nn.Linear(512, 128)
        self.Linear3 = nn.Linear(512, 256)
        self.Linear5 = nn.Linear(512, 1024)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def concat(self, upsampled, bypass):
        # 按维数1（列）拼接
        return torch.cat((upsampled, bypass), 1)

    def fusion(self, img, text):
        # skip = img
        global_attn = torch.sigmoid(self.avg(img) * text.unsqueeze(-1).unsqueeze(-1))
        img = img*global_attn
        B, C, H, W = img.shape
        img = img.reshape(B, C, H*W).transpose(-1, -2)
        text = text.unsqueeze(-1)
        attn = torch.sigmoid(img @ text)
        img = attn*img
        # img = skip + attn * img
        img = img.transpose(-1, -2).reshape(B, C, H, W)
        # img = img + skip*0.1
        return img

    def forward(self, x, text_tokens):
        # text Encode
        cond = self.clip_model.encode_text(text_tokens)
        cond = cond.to(torch.float32) # [B, 512]
        # cond1 = self.Linear1(cond)
        cond2 = self.Linear2(cond) # [B, 128]
        cond3 = self.Linear3(cond) # [B, 256]
        cond5 = self.Linear5(cond) # [B, 1024]

        # Encode
        encode_block1 = self.conv_encode1(x)             # [B, 64, 512, 512]
        encode_pool1 = self.conv_maxpool1(encode_block1) # [B, 64, 256, 256]
        encode_block2 = self.conv_encode2(encode_pool1)  # [B, 128, 256, 256]
        encode_pool2 = self.conv_maxpool2(encode_block2) # [B, 128, 128, 128]
        encode_block3 = self.conv_encode3(encode_pool2)  # [B, 256, 128, 128]
        encode_pool3 = self.conv_maxpool3(encode_block3) # [B, 256, 64, 64]
        encode_block4 = self.conv_encode4(encode_pool3)  # [B, 512, 64, 64]
        encode_pool4 = self.conv_maxpool4(encode_block4) # [B, 512, 32, 32]
        encode_block5 = self.conv_encode5(encode_pool4)  # [B, 1024, 32, 32]
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_block5)     # [B, 512, 64, 64]
        # Decode
        cat_layer1 = self.concat(bottleneck1, encode_block4)   # [B, 1024, 64, 64]
        cat_layer1 = self.fusion(cat_layer1, cond5)            # [B, 1024, 64, 64]
        decode_block1 = self.conv_decode1(cat_layer1)          # [B, 256, 128, 128]
        cat_layer2 = self.concat(decode_block1, encode_block3) # [B, 512, 128, 128]
        cat_layer2 = self.fusion(cat_layer2, cond)             # [B, 512, 128, 128]
        decode_block2 = self.conv_decode2(cat_layer2)          # [B, 128, 256, 256]
        cat_layer3 = self.concat(decode_block2, encode_block2) # [B, 256, 256, 256]
        cat_layer3 = self.fusion(cat_layer3, cond3)            # [B, 256, 256, 256]
        decode_block3 = self.conv_decode3(cat_layer3)          # [B, 64, 512, 512]
        cat_layer4 = self.concat(decode_block3, encode_block1) # [B, 128, 512, 512]
        cat_layer4 = self.fusion(cat_layer4, cond2)            # [B, 128, 512, 512]
        final_layer = self.final_layer(cat_layer4)             # [B, 1, 512, 512]
        final_layer = torch.sigmoid(final_layer)

        return final_layer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else cpu)
    image = torch.randn(1, 3, 512, 512).to(device)
    text = "Mito"
    text = clip.tokenize(text).to(device)
    model = Unet(3,1).to(device)
    pred = model(image, text)
    print(pred.shape)
