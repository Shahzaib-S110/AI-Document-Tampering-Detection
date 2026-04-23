# ============================================================
# 6.1 Test Set Evaluation — Pixel-Level Metrics
# ============================================================

@torch.no_grad()
def evaluate_test(model, loader, device):
    """Full evaluation on test set."""
    model.eval()
    all_probs = []
    all_masks = []
    all_labels = []

    for images, masks, labels in tqdm(loader, desc='Test Eval'):
        images = images.to(device)
        predictions = model(images)
        probs = torch.sigmoid(predictions)

        all_probs.append(probs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0) # (N, 1, H, W)
    all_masks = np.concatenate(all_masks, axis=0) # (N, 1, H, W)
    all_labels = np.array(all_labels)

    return all_probs, all_masks, all_labels

test_probs, test_masks, test_labels = evaluate_test(model, test_loader, DEVICE)
test_preds_binary = (test_probs > 0.5).astype(np.float32)

# Pixel-level metrics (flatten all pixels)
pred_flat = test_preds_binary.flatten()
mask_flat = test_masks.flatten()
prob_flat = test_probs.flatten()

eps = 1e-7
tp = (pred_flat * mask_flat).sum()
fp = (pred_flat * (1 - mask_flat)).sum()
fn = ((1 - pred_flat) * mask_flat).sum()
tn = ((1 - pred_flat) * (1 - mask_flat)).sum()

pixel_f1 = (2 * tp) / (2 * tp + fp + fn + eps)
pixel_iou = tp / (tp + fp + fn + eps)
pixel_dice = pixel_f1 # Dice = F1 for binary
pixel_precision = tp / (tp + fp + eps)
pixel_recall = tp / (tp + fn + eps)

# Pixel AUC (subsample for speed if needed)
n_pixels = len(prob_flat)
if n_pixels > 5_000_000:
    sample_idx = np.random.choice(n_pixels, 5_000_000, replace=False)
    pixel_auc = roc_auc_score(mask_flat[sample_idx], prob_flat[sample_idx])
else:
    pixel_auc = roc_auc_score(mask_flat, prob_flat) if mask_flat.sum() > 0 and (1-mask_flat).sum() > 0 else 0.0

print(f'{"="*60}')
print(f' PIXEL-LEVEL METRICS (Test Set)')
print(f'{"="*60}')
print(f' Pixel Precision: {pixel_precision:.4f}')
print(f' Pixel Recall: {pixel_recall:.4f}')
print(f' Pixel F1 (Dice): {pixel_f1:.4f}')
print(f' Pixel IoU: {pixel_iou:.4f}')
print(f' Pixel AUC: {pixel_auc:.4f}')
print(f'{"="*60}')


# ============================================================
# 6.2 Test Set Evaluation — Image-Level Classification
# ============================================================

# Derive image-level classification from masks:
# An image is classified as "tampered" if any predicted pixel > threshold
MASK_AREA_THRESHOLD = 100  # minimum number of tampered pixels to classify as tampered

image_pred_labels = []
image_pred_scores = []

for i in range(len(test_probs)):
    prob_map = test_probs[i, 0]  # (H, W)
    binary_map = (prob_map > 0.5).astype(np.float32)
    tampered_pixel_count = binary_map.sum()

    # Classification: tampered if enough pixels predicted as tampered
    pred_label = 1 if tampered_pixel_count >= MASK_AREA_THRESHOLD else 0
    image_pred_labels.append(pred_label)

    # Score: max probability in the mask (for ROC-AUC)
    image_pred_scores.append(prob_map.max())

image_pred_labels = np.array(image_pred_labels)
image_pred_scores = np.array(image_pred_scores)

# Classification metrics
cls_accuracy = accuracy_score(test_labels, image_pred_labels)
cls_report = classification_report(test_labels, image_pred_labels,
                                     target_names=['Authentic', 'Tampered'],
                                     output_dict=True)
cls_macro_f1 = f1_score(test_labels, image_pred_labels, average='macro')
cls_auc = roc_auc_score(test_labels, image_pred_scores) if len(np.unique(test_labels)) > 1 else 0.0

print(f'{"="*60}')
print(f'  IMAGE-LEVEL CLASSIFICATION (Test Set)')
print(f'{"="*60}')
print(f'  Test Accuracy:    {cls_accuracy:.4f}  ({cls_accuracy*100:.2f}%)')
print(f'  Macro F1:         {cls_macro_f1:.4f}')
print(f'  ROC-AUC:          {cls_auc:.4f}')
print(f'')
print(f'  Per-Class Results:')
print(f'  {"":>12} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
for cls_name in ['Authentic', 'Tampered']:
    r = cls_report[cls_name]
    print(f'  {cls_name:>12} {r["precision"]:>10.4f} {r["recall"]:>10.4f} {r["f1-score"]:>10.4f} {r["support"]:>10.0f}')
print(f'{"="*60}')

# Classification report (full)
print('\nFull Classification Report:')
print(classification_report(test_labels, image_pred_labels, target_names=['Authentic', 'Tampered']))


# ============================================================
# 6.3 Confusion Matrix and ROC Curve
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(test_labels, image_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentic', 'Tampered'],
            yticklabels=['Authentic', 'Tampered'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Confusion Matrix (Acc={cls_accuracy:.2%})')

# Print confusion details
tn, fp, fn, tp_cls = cm.ravel()
total = cm.sum()
print(f'Confusion Matrix:')
print(f' TN={tn}, FP={fp}, FN={fn}, TP={tp_cls}')
print(f' FP Rate: {fp/(tn+fp)*100:.1f}%')
print(f' FN Rate: {fn/(fn+tp_cls)*100:.1f}%')

# ROC Curve
if len(np.unique(test_labels)) > 1:
    fpr, tpr, thresholds = roc_curve(test_labels, image_pred_scores)
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {cls_auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve (AUC={cls_auc:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.suptitle(f'{VERSION} — Classification Performance', fontsize=14)
plt.tight_layout()
plt.show()
