import csv
from collections import defaultdict
import random
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.ticker import FuncFormatter
instance = 2

# 定义生成数据的函数
def generate_data():
    # 准备数据列表
    data_list = []
    
    # 生成数据
    for i in range(0, 4000):
        # 第一列数据
        passenger = i
        
        # start
        b = np.random.random()
        if  b > 0.55:
            start = random.randint(2, 4)
            
        else:
            start = random.randint(0, 2)
            
        time = random.randint(0,1000)

        # end
        end = start
        while end <= start:
            end = random.randint(1, 5)
        
        # time

       

        # Display the random integers
      
        # 将数据添加到列表
        data_list.append({
            'passenger': passenger,
            'start': start,
            'end': end,
            'time': time
        })
    
    # 将数据写入CSV文件
    with open('data_instance{}.csv'.format(instance), 'w', newline='') as csvfile:
        fieldnames = ['passenger', 'start', 'end', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入列名
        writer.writeheader()
        
        # 写入数据
        for data in data_list:
            writer.writerow(data)

def process_csv(instance):
    # 初始化字典，用于存储结果
    time_data = defaultdict(list)
    
    # 读取CSV文件
    with open('data_instance{}.csv'.format(instance), 'r', newline='') as csvfile:
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
# # Read the CSV file and parse the data


# Read data from CSV file
        # Read data from CSV file
    data = pd.read_csv('data_instance1.csv')
    
    data['start'] = data['start'] + 1

# Group the data by 'start' station and 'time' and count the number of passengers
    grouped_data = data.groupby(['start', 'time']).size().reset_index(name='passenger_count')

    # Create a meshgrid for interpolation
    start_values = np.unique(grouped_data['start'])
    time_values = np.unique(grouped_data['time'])
    start_mesh, time_mesh = np.meshgrid(start_values, time_values)

    # Calculate the cumulative number of passengers arriving at each station and time
    aa_data = grouped_data.groupby(['start', 'time'])['passenger_count'].sum().groupby('start').cumsum().reset_index(name='aa_passenger')

    # Interpolate the cumulative passenger counts using griddata
    aa_data['aa_passenger'] = aa_data['aa_passenger'].replace(np.nan, 0)

    aa_passenger_interp = griddata((aa_data['start'], aa_data['time']), aa_data['aa_passenger'],
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
    ax.set_xlabel('Start Station',fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    ax.set_zlabel('The number of passengers', fontsize=12)
    ax.set_title('Number of passengers boarding at the station', fontsize=15)

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
    data = pd.read_csv('data_instance1.csv')
    
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
    ax.set_title('Number of passengers alighting at the station', fontsize=15)

    # Set the X-axis interval to 1
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.margins(0)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    # Show the plot
    plt.show()
    plt.close()
    
    
def plot_flow3():

    # 读取CSV文件，假设列名是 "passenger_index", "start_station_index", "end_station_index", "arrival_time"
    df = pd.read_csv('data_instance2.csv', header=0)  # header=0 表示第一行是列名

    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['time'] = df['time'].astype(int)

    # 找出车站index和时间index的范围
    max_station_index = df['start'].max()
    max_time_index = df['time'].max()

    # 初始化累计乘客数量矩阵
    passenger_count_matrix = np.zeros((max_station_index + 1, max_time_index + 1))

    # 累计每个车站在每个时间点的乘客数量
    for index, row in df.iterrows():
        station_index = row['start']
        arrival_time = row['time']
        passenger_count_matrix[station_index, arrival_time] += 1

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    cumulative_passenger_matrix = np.cumsum(passenger_count_matrix, axis=1)

    # 绘制3D网状图
    X, Y = np.meshgrid(np.arange(passenger_count_matrix.shape[1]), np.arange(passenger_count_matrix.shape[0]))
    surf = ax.plot_surface(X, Y, cumulative_passenger_matrix, cmap='viridis')
    
        # 减少轴的边界空白
    ax.margins(x=0, y=0, z=0)

    # 调整视角以减少空白
    ax.view_init(elev=10, azim=10)

    # 自动调整坐标轴比例
    ax.autoscale(tight=True)
    
    

    # 创建FuncFormatter对象
    formatter_x = FuncFormatter(format_x)

    # 应用X轴的刻度格式化程序
    ax.xaxis.set_major_formatter(formatter_x)
    
    formatter_y = FuncFormatter(format_x)

    # 应用X轴的刻度格式化程序
    ax.xaxis.set_major_formatter(formatter_y)



    # 添加颜色条
    # cbar = fig.colorbar(surf, ax=ax, pad=0.1)
    # cbar.set_label('Cumulative Arriving Passenger',fontsize=12)

    # 设置坐标轴标签
    ax.set_xlabel('Time Index',fontsize=12)
    ax.set_ylabel('Station Index',fontsize=12)
    ax.set_zlabel('Cumulative Passenger Count',fontsize=12)
    
    
    # Set the X-axis interval to 1

    ax.margins(0)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    # 显示图形
    plt.show()
    
def format_x(x, pos):
        return int(x)  
    
def plot_flow4():

    # 读取CSV文件，假设列名是 "passenger_index", "start_station_index", "end_station_index", "arrival_time"
    df = pd.read_csv('data_instance2.csv', header=0)  # header=0 表示第一行是列名

    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['time'] = df['time'].astype(int)

    # 找出车站index和时间index的范围
    max_station_index = df['end'].max()
    max_time_index = df['time'].max()

    # 初始化累计乘客数量矩阵
    passenger_count_matrix = np.zeros((max_station_index + 1, max_time_index + 1))

    # 累计每个车站在每个时间点的乘客数量
    for index, row in df.iterrows():
        station_index = row['end']
        arrival_time = row['time']
        passenger_count_matrix[station_index, arrival_time] += 1

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    cumulative_passenger_matrix = np.cumsum(passenger_count_matrix, axis=1)

    # 绘制3D网状图
    X, Y = np.meshgrid(np.arange(passenger_count_matrix.shape[1]), np.arange(passenger_count_matrix.shape[0]))
    surf = ax.plot_surface(X, Y, cumulative_passenger_matrix, cmap='viridis')
    
    
        # 减少轴的边界空白
    ax.margins(x=0, y=0, z=0)

    # 调整视角以减少空白
    ax.view_init(elev=10, azim=10)

    # 自动调整坐标轴比例
    ax.autoscale(tight=True)


    # # 添加颜色条
    # cbar = fig.colorbar(surf, ax=ax, pad=0.1)
    # cbar.set_label('Cumulative Getting Off Passenger',fontsize=12)

    # 设置坐标轴标签
    ax.set_xlabel('Time Index',fontsize=12)
    ax.set_ylabel('Station Index',fontsize=12)
    ax.set_zlabel('Cumulative Passenger Count',fontsize=12)
    
    
    # Set the X-axis interval to 1

    ax.margins(0)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    # 显示图形
    plt.show()
    
    