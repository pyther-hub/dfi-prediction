import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Optional
from metrics import calculate_metrics
from tqdm import tqdm
import numpy as np

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_as: str,
    logger: Optional[object] = None,
    num_epochs: int = 25
) -> None:
    """
    Trains the model and evaluates its performance on the validation set.

    Parameters:
    ----------
    model : nn.Module
        The model to be trained.
    train_dataloader : DataLoader
        DataLoader for the training data.
    val_dataloader : DataLoader
        DataLoader for the validation data.
    criterion : nn.Module
        Loss function to optimize.
    optimizer : torch.optim.Optimizer
        Optimizer for the model parameters.
    save_as : str
        Filename prefix for saving model weights.
    logger : Optional[object], optional
        Logger object for logging metrics, by default None.
    num_epochs : int, optional
        Number of epochs to train, by default 25.

    Returns:
    -------
    None
    """
    # Set the device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the best validation Pearson correlation coefficient
    best_val_pearson_coeff = 0

    # Training loop over epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Variables to track losses and metrics during training
        train_running_loss = 0.0
        train_all_labels = []
        train_all_outputs = []

        # Training phase
        for train_inputs, train_labels in tqdm(train_dataloader, desc="Training"):
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            train_outputs = model(train_inputs)
            train_loss = criterion(train_outputs.squeeze(), train_labels.float())

            # Backward pass
            train_loss.backward()
            optimizer.step()

            # Accumulate training statistics
            train_running_loss += train_loss.item()
            train_all_labels.append(train_labels.cpu().numpy())
            train_all_outputs.append(train_outputs.squeeze().cpu().detach().numpy())

        # Calculate and print training metrics
        train_epoch_loss = train_running_loss / len(train_dataloader)
        train_all_labels = np.concatenate(train_all_labels)
        train_all_outputs = np.concatenate(train_all_outputs)

        train_mae, train_mse, train_pearson_corr = calculate_metrics(train_all_labels, train_all_outputs)
        print(f"Training Loss: {train_epoch_loss:.4f}")
        print(f"Training Pearson Correlation: {train_pearson_corr:.4f}")

        # Validation phase
        val_running_loss = 0.0
        val_all_labels = []
        val_all_outputs = []

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to compute gradients during validation
            for val_inputs, val_labels in tqdm(val_dataloader, desc="Validation"):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                # Forward pass
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs.squeeze(), val_labels.float())

                # Accumulate validation statistics
                val_running_loss += val_loss.item()
                val_all_labels.append(val_labels.cpu().numpy())
                val_all_outputs.append(val_outputs.squeeze().cpu().numpy())

        # Calculate and print validation metrics
        val_epoch_loss = val_running_loss / len(val_dataloader)
        val_all_labels = np.concatenate(val_all_labels)
        val_all_outputs = np.concatenate(val_all_outputs)

        val_mae, val_mse, val_pearson_corr = calculate_metrics(val_all_labels, val_all_outputs)
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"Validation Pearson Correlation: {val_pearson_corr:.4f}")

        # Save the model if the validation Pearson correlation improves
        if val_pearson_corr > best_val_pearson_coeff:
            best_val_pearson_coeff = val_pearson_corr
            torch.save(model.state_dict(), f'{save_as}_weights.pth')
            print(f"Model weights saved at epoch {epoch + 1}")

        # Log the metrics if a logger is provided
        if logger:
            logger.log({
                'train_loss': train_epoch_loss,
                'train_pearson_correlation': train_pearson_corr,
                'val_loss': val_epoch_loss,
                'val_mean_absolute_error': val_mae,
                'val_mean_squared_error': val_mse,
                'val_pearson_correlation': val_pearson_corr,
                'epoch_number': epoch + 1
            })

    # Log the best validation Pearson correlation coefficient
    if logger:
        logger.log({'best_val_pearson_coeff': best_val_pearson_coeff})

    # Final message indicating the end of training
    print("\nTraining and validation complete.")
