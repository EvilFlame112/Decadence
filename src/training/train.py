import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import multiprocessing

# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import the model
from models.model import InterpolationModel
from src.data_processing.dataset import VimeoDataset

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows

    # ----------------------------------------
    # GPU ENFORCEMENT & CONFIGURATION
    # ----------------------------------------
    def enforce_gpu():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA (GPU) is required but not available. Exiting.")
        device = torch.device("cuda")
        print(f"\nGPU detected: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")
        return device

    device = enforce_gpu()  # Will crash if GPU is unavailable
    torch.backends.cudnn.benchmark = True  # Optimize for GPU (if input sizes are fixed)

    # ----------------------------------------
    # HYPERPARAMETERS & MODEL SETUP
    # ----------------------------------------
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 0.0001

    # Initialize model and move to GPU immediately
    model = InterpolationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    # ----------------------------------------
    # DATA LOADER (WITH GPU-OPTIMIZED SETTINGS)
    # ----------------------------------------
    train_dataset = VimeoDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )

    # ----------------------------------------
    # TRAINING LOOP (GPU-ENFORCED)
    # ----------------------------------------
    print("\nTraining started (GPU-only mode)...")
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0.0

        # Progress bar with GPU memory monitoring
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for frame1, frame3, target in progress_bar:
            # Explicitly move data to GPU (redundant but safe)
            frame1 = frame1.to(device, non_blocking=True)
            frame3 = frame3.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward/backward pass
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            output = model(frame1, frame3)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Logging
            total_loss += loss.item()
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "GPU Mem": f"{torch.cuda.memory_allocated(device)/1e9:.2f}GB"
            })

        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        torch.cuda.empty_cache()  # Free unused GPU memory

    # Save model
    torch.save(model.state_dict(), "models/interpolator.pth")
    print("Training complete!")