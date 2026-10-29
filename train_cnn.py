"""Train the DermaScan CNN from an ImageFolder-style dataset.

Expected folders:
  data/train/class_a/*.jpg
  data/train/class_b/*.jpg
  data/val/class_a/*.jpg
  data/val/class_b/*.jpg

Class names are taken from the folder names. Use only a dataset you are licensed
and permitted to use. This script is for educational experimentation, not clinical
validation.
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, UnidentifiedImageError
from cnn_model import SkinCNN

IMAGE_SIZE = 224

class FolderDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str], augment: bool = False):
        self.root = root
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.augment = augment
        self.samples = []
        valid = {'.jpg', '.jpeg', '.png', '.webp'}
        for name in class_names:
            folder = root / name
            if not folder.exists():
                continue
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix.lower() in valid:
                    self.samples.append((p, self.class_to_idx[name]))
        if not self.samples:
            raise ValueError(f'No images found under {root}. Expected one subfolder per class.')

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            with Image.open(path) as im:
                image = im.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f'Could not read training image: {path}') from exc
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.augment and random.random() < 0.25:
            image = image.rotate(random.choice([90, 180, 270]))
        # Same RGB [0,1] preprocessing is used at inference.
        import numpy as np
        tensor = torch.from_numpy(np.asarray(image, dtype='float32').copy()).permute(2, 0, 1) / 255.0
        return tensor, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', help='Dataset root containing train/ and val/')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', default='models/skin_cnn.pth')
    args = parser.parse_args()

    torch.manual_seed(42)
    root = Path(args.data)
    train_root, val_root = root / 'train', root / 'val'
    if not train_root.exists() or not val_root.exists():
        raise SystemExit('Dataset folders missing. Create data/train/<class>/ and data/val/<class>/ first.')
    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise SystemExit('At least two class folders are required in data/train/.')
    train_ds = FolderDataset(train_root, classes, augment=True)
    val_ds = FolderDataset(val_root, classes, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkinCNN(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * images.size(0)
        model.eval(); correct = total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device))
                correct += (logits.argmax(1).cpu() == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / max(total, 1)
        print(f'Epoch {epoch+1}/{args.epochs} | train_loss={train_loss/len(train_ds):.4f} | val_accuracy={accuracy:.3f}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.cpu().state_dict(), 'classes': classes,
                        'image_size': IMAGE_SIZE, 'validation_accuracy': accuracy,
                        'note': 'Student CNN checkpoint; not clinically validated.'}, output)
            model.to(device)
            print(f'Saved best checkpoint: {output}')
    print('Training finished. Validation accuracy is a basic dataset metric, not evidence of clinical performance.')

if __name__ == '__main__': main()
