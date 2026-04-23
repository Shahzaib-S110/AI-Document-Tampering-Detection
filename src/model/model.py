Input: 384x384x3 (ELA map, ELA-normalized)
|
+-- ENCODER (ResNet-34, ImageNet pretrained)
|   +-- conv1:  7x7, 64, stride 2   -> 192x192x64    [FROZEN, BN UNFROZEN]
|   +-- pool:   3x3, stride 2       -> 96x96x64       [FROZEN]
|   +-- layer1: 3x[3x3, 64]         -> 96x96x64       [FROZEN, BN UNFROZEN]  [skip 1]
|   +-- layer2: 4x[3x3, 128]        -> 48x48x128      [FROZEN, BN UNFROZEN]  [skip 2]
|   +-- layer3: 6x[3x3, 256]        -> 24x24x256      [FROZEN, BN UNFROZEN]  [skip 3]
|   +-- layer4: 3x[3x3, 512]        -> 12x12x512      [FROZEN, BN UNFROZEN]  [skip 4]
|
+-- DECODER (UNet, TRAINABLE, lr=1e-3, ~500K params)
|   +-- up1: 12->24,  concat skip 3  -> 24x24
|   +-- up2: 24->48,  concat skip 2  -> 48x48
|   +-- up3: 48->96,  concat skip 1  -> 96x96
|   +-- up4: 96->192, concat skip 0  -> 192x192
|   +-- final: -> 384x384, 1x1 conv  -> sigmoid
|
Output: 384x384x1 (pixel probability)


### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Input | **ELA (Q=90)** | Tests forensic signal for pretrained localization |
| Encoder body | **FROZEN** | Conv weights still extract edges/textures from ELA |
| Encoder BN | **UNFROZEN** | Running stats adapt to ELA distribution (domain adaptation) |
| Normalization | **ELA-specific** | Computed from training set, not ImageNet |
| Decoder | UNet (TRAINABLE) | Skip connections preserve spatial detail |
| Loss | BCEDiceLoss | Handles class imbalance at pixel level |




# ============================================================
# 4.1 Build Model (Frozen Body + BN Unfrozen)
# ============================================================

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
    activation=None
)

# Step 1: Freeze ALL encoder parameters
for param in model.encoder.parameters():
    param.requires_grad = False

# Step 2: Unfreeze ONLY BatchNorm layers in encoder (domain adaptation)
bn_param_count = 0
for module in model.encoder.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        for param in module.parameters():
            param.requires_grad = True
            bn_param_count += param.numel()
        module.track_running_stats = True

model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
seghead_trainable = sum(p.numel() for p in model.segmentation_head.parameters() if p.requires_grad)

print(f'Model: UNet + {ENCODER} ({ENCODER_WEIGHTS}) \u2014 FROZEN BODY + BN UNFROZEN')
print(f'Total parameters:     {total_params:>12,}')
print(f'Trainable parameters: {trainable_params:>12,}')
print(f'  Encoder BN params:  {encoder_trainable:>12,}  (BatchNorm only, lr={LEARNING_RATE})')
print(f'  Decoder:            {decoder_trainable:>12,}  (lr={LEARNING_RATE})')
print(f'  Segmentation head:  {seghead_trainable:>12,}  (lr={LEARNING_RATE})')
print(f'Frozen parameters:    {frozen_params:>12,}  (all conv/fc weights)')
print(f'Trainable ratio:      {trainable_params/total_params*100:.1f}%')
print(f'Data:param ratio:     1 : {trainable_params/len(train_dataset):.0f}')
