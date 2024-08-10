from models.imagenet.resnet import ResNetDCT_Upscaled_Static
model = ResNetDCT_Upscaled_Static(channels=192, pretrained=True)
print(model)