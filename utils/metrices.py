from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def calculate_metrics(labels, outputs):
	mean_absolute_error_value = mean_absolute_error(labels, outputs)
	mean_squared_error_value = mean_squared_error(labels, outputs)
	pearson_correlation, _ = pearsonr(labels, outputs)  # Use pearsonr to calculate Pearson correlation
	
	return mean_absolute_error_value, mean_squared_error_value, pearson_correlation