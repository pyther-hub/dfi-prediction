import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import torch



def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, logger=None, num_epochs=25):
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.to(device)
	
	for epoch in range(num_epochs):
		model.train()  # Set model to training mode
		train_running_loss = 0.0
		train_all_labels = []
		train_all_outputs = []
		
		for train_inputs, train_labels in tqdm(train_dataloader, desc="Training"):
			train_inputs = train_inputs.to(device)
			train_labels = train_labels.to(device)

			optimizer.zero_grad()

			train_outputs = model(train_inputs)
			train_loss = criterion(train_outputs.squeeze(), train_labels.float())
						
			train_loss.backward()
			optimizer.step()

			train_running_loss += train_loss.item()
			train_all_labels.append(train_labels.cpu().numpy())
			train_all_outputs.append(train_outputs.squeeze().cpu().detach().numpy())
		
		train_epoch_loss = train_running_loss / len(train_dataloader)
		train_all_labels = np.concatenate(train_all_labels)
		train_all_outputs = np.concatenate(train_all_outputs)

		train_mean_absolute_error, train_mean_squared_error, train_pearson_correlation = calculate_metrics(train_all_labels, train_all_outputs)

		print(f'Training Loss: {train_epoch_loss:.4f}')
		print(f'Training Pearson Correlation: {train_pearson_correlation:.4f}')

		# Validation phase
		val_running_loss = 0.0
		val_all_labels = []
		val_all_outputs = []
		model.eval()  # Set model to evaluate mode

		with torch.no_grad():
			for val_inputs, val_labels in tqdm(val_dataloader, desc="Validation"):
				val_inputs = val_inputs.to(device)
				val_labels = val_labels.to(device)
				# Forward pass
				val_outputs = model(val_inputs)
				val_loss = criterion(val_outputs.squeeze(), val_labels.float())

				# Statistics
				val_running_loss += val_loss.item()
				val_all_labels.append(val_labels.cpu().numpy())
				val_all_outputs.append(val_outputs.squeeze().cpu().numpy())
		
		val_epoch_loss = val_running_loss / len(val_dataloader)
		val_all_labels = np.concatenate(val_all_labels)
		val_all_outputs = np.concatenate(val_all_outputs)

		# Calculate validation metrics
		val_mean_absolute_error, val_mean_squared_error, val_pearson_correlation = calculate_metrics(val_all_labels, val_all_outputs)

		# Logging
		if logger is not None:
			logger.log({
				'train_loss': train_epoch_loss,
				'val_loss': val_epoch_loss,
				'val_mean_absolute_error': val_mean_absolute_error,
				'val_mean_squared_error': val_mean_squared_error,
				'val_pearson_correlation': val_pearson_correlation,
				'epoch_number': epoch + 1  # Epoch number starting from 1
			})

		print(f'Validation Loss: {val_epoch_loss:.4f}')
		print(f'Validation Pearson Correlation: {val_pearson_correlation:.4f}')

	print('Training and Validation complete')
