# ============================================================
# 5.1 Loss, Optimizer, Scheduler
# ============================================================

bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def criterion(pred, target):
    return bce_loss_fn(pred, target) + dice_loss_fn(pred, target)

# Single optimizer — all trainable params at same LR (vR.P.7)
trainable_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params_list, lr=LEARNING_RATE, weight_decay=1e-5)

print(f'Optimizer: Adam (single LR)')
print(f'  Trainable params: {sum(p.numel() for p in trainable_params_list):,}')
print(f'  LR: {LEARNING_RATE}')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ============================================================
# 5.2 Training and Validation Functions (AMP-enabled)
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    for images, masks, labels in tqdm(loader, desc='Train', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_masks = []
    for images, masks, labels in tqdm(loader, desc='Val', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, masks)
        total_loss += loss.item()
        num_batches += 1
        probs = torch.sigmoid(predictions.float())
        all_preds.append(probs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
    avg_loss = total_loss / num_batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    binary_preds = (all_preds > 0.5).astype(np.float32)
    pred_flat = binary_preds.flatten()
    mask_flat = all_masks.flatten()
    eps = 1e-7
    tp = (pred_flat * mask_flat).sum()
    fp = (pred_flat * (1 - mask_flat)).sum()
    fn = ((1 - pred_flat) * mask_flat).sum()
    pixel_f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    pixel_iou = tp / (tp + fp + fn + eps)
    return avg_loss, pixel_f1, pixel_iou

print('Training functions ready (AMP-enabled).')





# ============================================================
# 5.3 Training Loop (with checkpoint save/resume + AMP)
# ============================================================

scaler = GradScaler('cuda')

history = {
    'train_loss': [], 'val_loss': [], 'val_pixel_f1': [], 'val_pixel_iou': [],
    'lr': []
}

best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0
best_model_state = None
start_epoch = 1

if os.path.exists(CHECKPOINT_PATH):
    print(f'Checkpoint found: {CHECKPOINT_PATH}')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    best_epoch = ckpt['best_epoch']
    patience_counter = ckpt['patience_counter']
    history = ckpt['history']
    if ckpt.get('best_model_state') is not None:
        best_model_state = ckpt['best_model_state']
    print(f'  Resuming from epoch {start_epoch} (best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})')
else:
    print('No checkpoint found \u2014 starting fresh.')

print(f'Starting training: epochs {start_epoch}-{EPOCHS}, patience={PATIENCE}')
print(f'LR: {LEARNING_RATE} | Input: ELA (Q={ELA_QUALITY}) | AMP: Enabled')
print(f'{"="*80}')

for epoch in range(start_epoch, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]['lr']
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
    val_loss, val_f1, val_iou = validate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_pixel_f1'].append(val_f1)
    history['val_pixel_iou'].append(val_iou)
    history['lr'].append(current_lr)
    improved = ''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        improved = ' *'
    else:
        patience_counter += 1
    print(f'Epoch {epoch:>2}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
          f'Pixel F1: {val_f1:.4f} | IoU: {val_iou:.4f} | LR: {current_lr:.2e}{improved}')
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
        'patience_counter': patience_counter, 'best_model_state': best_model_state,
        'history': history,
    }, CHECKPOINT_PATH)
    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}')
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model = model.to(DEVICE)
    print(f'\nRestored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})')
else:
    print('\nNo improvement during training \u2014 using final weights')

print(f'{"="*80}')
print(f'Training complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}')
