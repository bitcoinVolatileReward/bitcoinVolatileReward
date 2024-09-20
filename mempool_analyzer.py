import requests
from datetime import datetime
import os
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression

def monotonic_poly(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def curve_func(x, a, b, c):
    # return a * (x**2) + b * (x) + c * (x**0.5)
    return a * (x**b) + c
#     return a * np.log(x + b) + c
    # return a * x + b

# Function to convert date string to Unix epoch time
def date_to_epoch(date_str):
    dt = datetime.strptime(date_str, "%d/%m/%Y_%H_%M_%S")
    return int(dt.timestamp())


def download_data(start_date_str, end_date_str, file_name):
    # Convert dates to Unix epoch time
    s = date_to_epoch(start_date_str)
    e = date_to_epoch(end_date_str)

    # Construct the URL using the Unix epoch times
    url = f"https://johoe.jochen-hoenicke.de/queue/db.php?s={s}&e={e}&i=1"



    # Send a GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the relevant JSON part without the 'call' wrapper
        start_index = response.text.find("[[")
        end_index = response.text.rfind("]]") + 2
        json_data = response.text[start_index:end_index]

        # Reformat the data string to ensure it is in valid JSON format
        # json_data = "[[" + json_data + "]]"

        # Parse the JSON data
        try:
            data = json.loads(json_data)

            # Save the extracted JSON data to a file
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file)

            print(f"Data has been saved to {file_name}")
        except json.JSONDecodeError as e:
            print("Error decoding JSON data:", e)
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")


class Mempool_data():
    def __init__(self, Mempool_args):
        self.start_date_str = Mempool_args.start_date_str
        self.end_date_str = Mempool_args.end_date_str
        self.old_range_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 140, 170, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1700, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
        self.Mempool_args = Mempool_args
    def extract_mempool_data(self):
        # Read the JSON file
        checkpoint_dir = 'Mempool Data'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        file_name = f"data_{self.start_date_str}_{self.end_date_str}.json"
        file_name = file_name.replace("/", "-").replace(":", "-")
        file_name = os.path.join(checkpoint_dir, file_name)
        if not os.path.isfile(file_name):
            download_data(self.start_date_str, self.end_date_str, file_name)
        
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)

        modified_weights = []
        block_mined_timestamps = []
        
        time_last_block = 0
        weight_increase_rate_data = []
        weight_data = []
        base_fee_range_index = []
        for index in range(1, len(data)):
            element = data[index]
            weight_array = element[2]
            sum = 0
            for i in reversed(range(len(weight_array))):
                sum += weight_array[i]
                if sum > 10**6:
                    base_fee_range_index.append(i)
                    break
        base_fee_index = int(np.floor(np.min(base_fee_range_index)))
        base_fee_range = self.old_range_array[base_fee_index]
        print('Base fee range:', base_fee_range)
        self.new_range_array = np.arange(base_fee_range, base_fee_range + self.Mempool_args.N_memPool_section * self.Mempool_args.sat_per_byte_range_length,
                                         self.Mempool_args.sat_per_byte_range_length)
        weight_last_block = np.zeros(len(self.new_range_array))
        print(self.new_range_array)
        for index in range(1, len(data)):
            element = data[index]
            weight_array = element[2]
            agg_weights = [0] * len(self.new_range_array)

            for i in range(len(self.new_range_array)):
                for j in range(1, len(self.old_range_array)):
                    if self.old_range_array[j] > self.new_range_array[i]:
                        start = self.old_range_array[j - 1]
                        start_index = j-1
                        break
                if i < len(self.new_range_array) - 1:
                    for j in range(len(self.old_range_array)-2, -1, -1):
                        if self.old_range_array[j] < self.new_range_array[i+1]:
                            end = self.old_range_array[j + 1]
                            end_index = j + 1
                            break
                if i == len(self.new_range_array) - 1:
                    agg_weights[i] = (self.old_range_array[start_index + 1] - self.new_range_array[i]) / (self.old_range_array[start_index + 1] - start) * weight_array[start_index] if start_index < len(self.old_range_array) - 1 else 0
                    for j in range(start_index + 1, len(self.old_range_array)):
                        agg_weights[i] += weight_array[j]
                elif start_index + 1 == end_index:
                    agg_weights[i] = (self.new_range_array[i+1] - self.new_range_array[i])/(end - start) * weight_array[start_index]
                else:
                    agg_weights[i] = (self.old_range_array[start_index + 1] - self.new_range_array[i]) / (self.old_range_array[start_index + 1] - start) * weight_array[start_index] + \
                                     (self.new_range_array[i+1] - self.old_range_array[end_index - 1]) / (
                                                 end - self.old_range_array[end_index - 1]) * weight_array[end_index-1]
                    for j in range(start_index + 1, end_index-1):
                        agg_weights[i] += weight_array[j]
                # print(start, self.new_range_array[i], end, np.sum(agg_weights), np.sum(weight_array[base_fee_index:]))

            modified_weights.append(agg_weights)

            if index == 1:
                weight_last_block = modified_weights[-1]
            if np.sum(weight_array) < np.sum(data[index-1][2]):
                block_mined_timestamps.append(data[index-1][0])
                unix_timestamp = data[index - 1][0]
                struct_time = time.localtime(unix_timestamp)
                # human_readable_time = time.strftime('%Y-%m-%d %H:%M:%S', struct_time)
                # # print(data[index-1][0], human_readable_time, np.sum(np.array(weight_array) - np.array(data[index-1][2]) <= 0))
                time_last_block = 0
                weight_last_block = modified_weights[-1]
            elif index > 1:
                time_last_block += 1
                weight_increase_rate_data.append([time_last_block])
                weight_data.append([time_last_block])
                for i in range(len(self.new_range_array)):
                    weight_increase_rate_data[-1].append(modified_weights[-1][i]-modified_weights[-2][i])
                    weight_data[-1].append(max(0, modified_weights[-1][i] - weight_last_block[i]))

        total_weights = np.zeros(len(self.new_range_array))
        for i in range(len(modified_weights)):
            total_weights = np.add(total_weights, modified_weights[i])

        new_weight = []
        rate_matrix = np.zeros((len(self.new_range_array)+1, len(weight_increase_rate_data)))
        slope = []
        intercept = []
        sum = 0
        for i in range(len(self.new_range_array)+1):
            for j in range(len(weight_increase_rate_data)):
                rate_matrix[i, j] = weight_increase_rate_data[j][i]
            if i > 0:
                print(
                    f'Average weight increase per 10 minutes for range[{self.new_range_array[i - 1]},{self.new_range_array[i] if i < len(self.new_range_array) else 1000}) (vMB): {10 * np.average(rate_matrix[i]) / 1e6}')
                new_weight.append(np.average(rate_matrix[i]) / 1e6)
                sum += np.average(rate_matrix[i])
                x = np.array(rate_matrix[0]).reshape(-1, 1)
                y = np.array(np.clip(rate_matrix[i]/1e6, 0, None)).reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                y_pred = model.predict(x)
                slope.append(model.coef_[0][0])
                intercept.append(model.intercept_[0])
                # Plot the regression line
                # plt.plot(x, y_pred, label=f'fee per weight (sat/Byte): [{self.new_range_array[i-1]},{self.new_range_array[i] if i<len(self.new_range_array) else 1000})')
        print('Total weight increase per 10 minutes:', 10*sum/1e6)
#         print('Weight  increase rate', new_weight)
        # average_size = np.divide(total_weights, 1e6 * len(modified_weights))
        # print('Average size', list(average_size))
        
        new_weight = []
        rate_matrix = np.zeros((len(self.new_range_array)+1, len(weight_data)))
        coef = []
        noise_std = []
        
        sum = 0
        for i in range(len(self.new_range_array)+1):
            for j in range(len(weight_data)):
                rate_matrix[i, j] = weight_data[j][i]
            if i > 0:
                new_weight.append(np.average(rate_matrix[i]) / 1e6)
                sum += np.average(rate_matrix[i])
                x = np.array(rate_matrix[0]).reshape(-1, 1)
                y = np.array(np.clip(rate_matrix[i]/ 1e6, 0, None)).reshape(-1, 1)
                # monotonic curve
                x = np.array(rate_matrix[0])
                y = np.array(np.clip(rate_matrix[i] / 1e6, 0, None))
                sorted_indices = np.argsort(x)
                x_sorted = x[sorted_indices]
                y_sorted = y[sorted_indices]
                popt, pcov = curve_fit(curve_func, x_sorted, y_sorted, method='trf', bounds=(0, np.inf))
                coef.append(list(popt))
                y_pred = curve_func(x_sorted, *popt)
                std = np.std(y_sorted - y_pred)
                noise_std.append(std)
                # Generate a range of x values for plotting the fitted curve
                x_curve = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                y_curve = curve_func(x_curve, *popt)
                y_curve_noisy = y_curve + np.random.normal(0, std, x_curve.shape[0])
                plt.scatter(x_sorted, y_sorted, label=f'fee per weight (sat/Byte): [{self.new_range_array[i - 1]},{self.new_range_array[i] if i < len(self.new_range_array) else 1000})')
                # plt.scatter(x_curve, y_curve_noisy,
                #             label=f'fee per weight (sat/Byte): [{self.new_range_array[i - 1]},{self.new_range_array[i] if i < len(self.new_range_array) else 1000})')
                plt.plot(x_curve, y_curve,
                                  label=f'range (sat/vByte): [{self.new_range_array[i - 1]},{self.new_range_array[i] if i < len(self.new_range_array) else "\u221E"})')
        
#         print('Coefficients', coef)
#         print('noise_std', noise_std)
        plt.xlabel('Block generation time (minutes)')
        plt.ylabel('Total weight of transactions (vBytes)')
        plt.legend(fontsize=8)
        plt.grid()
#         plt.savefig('Fig6.png', dpi=300)
        plt.show()
        return coef, noise_std, base_fee_range



