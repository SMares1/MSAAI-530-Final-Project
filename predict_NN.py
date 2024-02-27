import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Standardize Training data
def get_standard_data(df_x, df_y):
	for col in df_x.columns:
		top = np.max(df_x[col])
		bot = np.min(df_x[col])

		df_x.loc[:, col] = (df_x.loc[:, col] - bot) / (top - bot)  # Normalize data to have max and min of 1-0

	for col in df_y.columns:
		top = np.max(df_y[col])
		bot = np.min(df_y[col])

		df_y.loc[:, col] = (df_y.loc[:, col] - bot) / (top - bot)  # Normalize data to have max and min of 1-0

	return df_x, df_y

def unnormalize_data(data, original_data):
	unnormalized_data = data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)
	return unnormalized_data

if __name__ == "__main__":
	file_path = "Data/filtered_ecosmart_pot_data.csv"

	start_date = "2016-08-01"
	end_date = "2016-08-31"

	df = pd.read_csv(file_path)

	df['date'] = pd.to_datetime(df['startdate'])

	df_filtered_index = df[(df['date'] >= start_date) & (df['date'] <= end_date)].index 

	features = ["temp_stress", "wind_stress"]
	target = ["plant_stress_indicator"]

	# Grab the data only for that date range.
	X = df[features]
	y = df[target]

	norm_X, norm_y = get_standard_data(X, y)

	example_X = norm_X.loc[df_filtered_index]
	example_y = norm_y.loc[df_filtered_index]

	model = tf.keras.models.load_model("Models/model_64x1")
	print(model.summary())
	print(f"Example X shape: {example_X.shape}")
	print(f"Example y shape: {example_y.shape}")

	pred_y = model.predict(example_X)

	unnorm_pred_y = unnormalize_data(pred_y, y)
	unnorm_example_y = unnormalize_data(example_y, y)

	# Plot model results for now.
	dates = df['date'].loc[df_filtered_index]

	daily_data = pd.DataFrame({'Date': dates, 'Pred_Y': unnorm_pred_y.flatten(), 'Example_Y': unnorm_example_y.values.flatten()})
	# Group by date and calculate the mean, min, max, upper quartile, and lower quartile
	daily_data = daily_data.groupby('Date').agg({'Pred_Y': ['mean', 'min', 'max', lambda x: np.percentile(x, 75), lambda x: np.percentile(x, 25)], 'Example_Y': 'mean'}).reset_index()

	# Rename the columns
	daily_data.columns = ['Date', 'Pred_psi_Mean', 'Pred_psi_Min', 'Pred_psi_Max', 'Pred_psi_Upper_Quartile', 'Pred_psi_Lower_Quartile', 'Truth_psi_Mean']

	plt.scatter(daily_data['Date'], daily_data['Pred_psi_Mean'], label="model_pred_Mean")
	plt.scatter(daily_data['Date'], daily_data['Pred_psi_Max'], label="model_pred_Max")
	plt.scatter(daily_data['Date'], daily_data['Pred_psi_Min'], label="model_pred_Min")
	plt.legend()
	plt.show()

	# Save daily_data as CSV
	daily_data.to_csv("Data/daily_data.csv", index=False)

