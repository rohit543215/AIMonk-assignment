import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataloaders
from model import build_model, unfreeze_model, get_pos_weights


CONFIG = {
    'labels_file': 'data/labels.txt',
    'images_dir': 'data/images/images',
    'batch_size': 32,
    'num_workers': 0,
    'val_split': 0.2,
    'seed': 42,
    
    'stage1_epochs': 5,      # Increased for better learning
    'stage1_lr': 1e-3,
    
    'stage2_epochs': 15,     # Increased for better convergence
    'stage2_lr': 1e-4,       # Higher LR for better learning
    
    'num_classes': 4,
    'dropout': 0.3,          # Reduced to allow more learning
    'weight_decay': 1e-4,
    'early_stop_patience': 5, # More patience
    'save_path': 'model.pth',
}


def masked_bce_loss(criterion, outputs, labels, masks):
    """Computes BCE loss only on known (non-NA) positions."""
    loss = criterion(outputs, labels)
    loss = loss * masks
    return loss.sum() / masks.sum()


def plot_loss_curve(iterations, losses):
    """Plots training loss curve as per assignment requirements."""
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, losses, color='steelblue', linewidth=1.5)
    plt.xlabel('iteration_number')
    plt.ylabel('training_loss')
    plt.title('Aimonk_multilabel_problem')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    print("Loss curve saved to loss_curve.png")


def train_one_epoch(model, loader, criterion, optimizer, device, iteration_count, all_iterations, all_losses):
    """Trains for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    for images, labels, masks in loader:
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)
        
        outputs = model(images)
        loss = masked_bce_loss(criterion, outputs, labels, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iteration_count += 1
        all_iterations.append(iteration_count)
        all_losses.append(loss.item())
        epoch_loss += loss.item()
    
    return iteration_count, epoch_loss / len(loader)


def compute_accuracy(outputs, labels, masks, threshold=0.5):
    """
    Computes accuracy metrics for multilabel classification.
    Only considers positions where mask=1 (known labels).
    """
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).float()
    
    # Only evaluate where we have ground truth (mask=1)
    valid_positions = masks == 1
    
    if valid_positions.sum() == 0:
        return 0.0, 0.0, [0.0] * 4
    
    # Hamming accuracy (per-label accuracy)
    correct = (preds == labels) * valid_positions
    hamming_acc = correct.sum().item() / valid_positions.sum().item()
    
    # Exact match accuracy (all labels correct per sample)
    # For each sample, check if all valid positions match
    exact_matches = 0
    total_samples = 0
    for i in range(len(labels)):
        sample_mask = valid_positions[i]
        if sample_mask.sum() > 0:
            sample_correct = (preds[i] == labels[i]) * sample_mask
            if sample_correct.sum() == sample_mask.sum():
                exact_matches += 1
            total_samples += 1
    
    exact_acc = exact_matches / total_samples if total_samples > 0 else 0.0
    
    # Per-attribute accuracy
    attr_accs = []
    for attr_idx in range(4):
        attr_mask = valid_positions[:, attr_idx]
        if attr_mask.sum() > 0:
            attr_correct = (preds[:, attr_idx] == labels[:, attr_idx]) * attr_mask
            attr_acc = attr_correct.sum().item() / attr_mask.sum().item()
            attr_accs.append(attr_acc)
        else:
            attr_accs.append(0.0)
    
    return hamming_acc, exact_acc, attr_accs


def validate(model, loader, criterion, device):
    """Validates the model with loss and accuracy metrics."""
    model.eval()
    epoch_loss = 0.0
    all_hamming = []
    all_exact = []
    all_attr_accs = [[] for _ in range(4)]
    
    with torch.no_grad():
        for images, labels, masks in loader:
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            outputs = model(images)
            loss = masked_bce_loss(criterion, outputs, labels, masks)
            epoch_loss += loss.item()
            
            # Compute accuracy
            hamming_acc, exact_acc, attr_accs = compute_accuracy(outputs, labels, masks)
            all_hamming.append(hamming_acc)
            all_exact.append(exact_acc)
            for i, acc in enumerate(attr_accs):
                all_attr_accs[i].append(acc)
    
    avg_loss = epoch_loss / len(loader)
    avg_hamming = sum(all_hamming) / len(all_hamming)
    avg_exact = sum(all_exact) / len(all_exact)
    avg_attr_accs = [sum(accs) / len(accs) for accs in all_attr_accs]
    
    return avg_loss, avg_hamming, avg_exact, avg_attr_accs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        labels_file=CONFIG['labels_file'],
        images_dir=CONFIG['images_dir'],
        val_split=CONFIG['val_split'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        seed=CONFIG['seed'],
    )
    
    # Model with dropout
    model = build_model(num_classes=CONFIG['num_classes'], pretrained=True, dropout=CONFIG['dropout'])
    model = model.to(device)
    
    # Loss with class weights
    pos_weights = get_pos_weights(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
    
    # Tracking
    all_iterations = []
    all_losses = []
    iteration_count = 0
    best_val_loss = float('inf')
    best_hamming = 0.0
    best_exact = 0.0
    best_attr_accs = [0.0] * 4
    patience_counter = 0
    
    # STAGE 1: Train classifier head only
    print("\n" + "="*50)
    print("STAGE 1: Training classifier head")
    print("="*50)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['stage1_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    for epoch in range(1, CONFIG['stage1_epochs'] + 1):
        iteration_count, train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            iteration_count, all_iterations, all_losses
        )
        val_loss, val_hamming, val_exact, val_attr_accs = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{CONFIG['stage1_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Hamming Acc: {val_hamming:.3f} | Exact Acc: {val_exact:.3f}")
        print(f"  Per-Attr Acc: [Attr1: {val_attr_accs[0]:.3f}, Attr2: {val_attr_accs[1]:.3f}, "
              f"Attr3: {val_attr_accs[2]:.3f}, Attr4: {val_attr_accs[3]:.3f}]")
    
    # STAGE 2: Fine-tune with partial unfreezing
    print("\n" + "="*50)
    print("STAGE 2: Fine-tuning (partial unfreeze)")
    print("="*50)
    
    model = unfreeze_model(model, freeze_layers=5)  # Keep early layers frozen
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['stage2_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    for epoch in range(1, CONFIG['stage2_epochs'] + 1):
        iteration_count, train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            iteration_count, all_iterations, all_losses
        )
        val_loss, val_hamming, val_exact, val_attr_accs = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{CONFIG['stage2_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Hamming: {val_hamming:.3f} | Exact: {val_exact:.3f}", end='')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping with model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hamming = val_hamming
            best_exact = val_exact
            best_attr_accs = val_attr_accs
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(" ✓ Saved")
        else:
            patience_counter += 1
            print()
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"Early stopping triggered (patience={CONFIG['early_stop_patience']})")
                break
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("Best Validation Metrics:")
    print(f"  Loss: {best_val_loss:.4f}")
    print(f"  Hamming Accuracy: {best_hamming:.3f} ({best_hamming*100:.1f}%)")
    print(f"  Exact Match Accuracy: {best_exact:.3f} ({best_exact*100:.1f}%)")
    print("  Per-Attribute Accuracy:")
    print(f"    Attr1: {best_attr_accs[0]:.3f} ({best_attr_accs[0]*100:.1f}%)")
    print(f"    Attr2: {best_attr_accs[1]:.3f} ({best_attr_accs[1]*100:.1f}%)")
    print(f"    Attr3: {best_attr_accs[2]:.3f} ({best_attr_accs[2]*100:.1f}%)")
    print(f"    Attr4: {best_attr_accs[3]:.3f} ({best_attr_accs[3]*100:.1f}%)")
    print("="*60)
    print(f"Model saved to: {CONFIG['save_path']}")
    
    plot_loss_curve(all_iterations, all_losses)


if __name__ == '__main__':
    main()
