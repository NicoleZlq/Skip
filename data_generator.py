import csv
from collections import defaultdict
import random
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata








# 定义生成数据的函数
def generate_data():
    # 准备数据列表
    data_list = []
    
    # 生成数据
    for i in range(0, 8000):
        # 第一列数据
        passenger = i
        
        # start
        if i > 6000:
            start = random.randint(0, 4)
            
        else:
            start = random.randint(0, 2)

        # end
        end = start
        while end <= start:
            end = random.randint(1, 5)
        
        # time

        time = random.randint(0,1000)

        # Display the random integers
      
        # 将数据添加到列表
        data_list.append({
            'passenger': passenger,
            'start': start,
            'end': end,
            'time': time
        })
    
    # 将数据写入CSV文件
    with open('data.csv', 'w', newline='') as csvfile:
        fieldnames = ['passenger', 'start', 'end', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入列名
        writer.writeheader()
        
        # 写入数据
        for data in data_list:
            writer.writerow(data)

def process_csv():
    # 初始化字典，用于存储结果
    time_data = defaultdict(list)
    
    # 读取CSV文件
    with open('data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # 遍历CSV文件中的每一行
        for row in reader:
            start = int(row['start'])
            end = int(row['end'])
            time = int(row['time'])
            
            # 构建一个包含time, start, end的元组，并将其添加到time_data字典中
            data_tuple = (start, end)
            time_data[time].append(data_tuple)

    

    return time_data

# Function to read the CSV file and parse the data
def read_csv_data(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip the header row
        data = [(int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in reader]
    return data


def plot_flow1():
# Read the CSV file and parse the data


# Read data from CSV file
        # Read data from CSV file
    data = pd.read_csv('data.csv')
    
    data['start'] = data['start'] + 1

# Group the data by 'start' station and 'time' and count the number of passengers
    grouped_data = data.groupby(['start', 'time']).size().reset_index(name='passenger_count')

    # Create a meshgrid for interpolation
    start_values = np.unique(grouped_data['start'])
    time_values = np.unique(grouped_data['time'])
    start_mesh, time_mesh = np.meshgrid(start_values, time_values)

    # Calculate the cumulative number of passengers arriving at each station and time
    cumulative_data = grouped_data.groupby(['start', 'time'])['passenger_count'].sum().groupby('start').cumsum().reset_index(name='cumulative_passenger')

    # Interpolate the cumulative passenger counts using griddata

    cumulative_passenger_interp = griddata((cumulative_data['start'], cumulative_data['time']), cumulative_data['cumulative_passenger'],
                                        (start_mesh, time_mesh), method='cubic')
    


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(start_mesh, time_mesh, cumulative_passenger_interp, 
                       cmap='gnuplot', edgecolor='none')

    # Add a color bar which shows the cumulative number of passengers
    # The color bar will use the normalization we set


    # Set labels for the axes and title of the plot
    ax.set_xlabel('Start Station',fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    ax.set_zlabel('The number of passengers', fontsize=12)
    ax.set_title('The total number of passengers arriving at the station', fontsize=15)

    # Set the X-axis interval to 1
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.margins(0)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    # Show the plot
    plt.show()
    
    plt.close()
    
def plot_flow2():
# Read the CSV file and parse the data


# Read data from CSV file
        # Read data from CSV file
    data = pd.read_csv('data.csv')
    
    data['end'] = data['end'] + 1

# Group the data by 'start' station and 'time' and count the number of passengers
    grouped_data = data.groupby(['end', 'time']).size().reset_index(name='passenger_count')

    # Create a meshgrid for interpolation
    start_values = np.unique(grouped_data['end'])
    time_values = np.unique(grouped_data['time'])
    start_mesh, time_mesh = np.meshgrid(start_values, time_values)

    # Calculate the cumulative number of passengers arriving at each station and time
    aa_data = grouped_data.groupby(['end', 'time'])['passenger_count'].sum().groupby('end').cumsum().reset_index(name='aa_passenger')

    # Interpolate the cumulative passenger counts using griddata
    aa_data['aa_passenger'] = aa_data['aa_passenger'].replace(np.nan, 0)

    aa_passenger_interp = griddata((aa_data['end'], aa_data['time']), aa_data['aa_passenger'],
                                        (start_mesh, time_mesh), method='cubic')
    


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(start_mesh, time_mesh, aa_passenger_interp, 
                       cmap='gnuplot', vmin=np.nanmin(aa_passenger_interp),vmax=np.nanmax(aa_passenger_interp))

    # Add a color bar which shows the cumulative number of passengers
    # The color bar will use the normalization we set


    # Set labels for the axes and title of the plot
    ax.set_xlabel('End Station',fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    ax.set_zlabel('The number of passengers', fontsize=12)
    ax.set_title('The total number of passengers getting off at the station', fontsize=15)

    # Set the X-axis interval to 1
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.margins(0)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    # Show the plot
    plt.show()
    plt.close()
