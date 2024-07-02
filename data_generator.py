import csv
from collections import defaultdict
import random
import csv



# 定义生成数据的函数
def generate_data():
    # 准备数据列表
    data_list = []
    
    # 生成数据
    for i in range(0, 100):
        # 第一列数据
        passenger = i
        
        # 第二列数据，从1到5的随机整数
        start = random.randint(1, 5)
        
        # 第三列数据，从2到6的随机整数，且大于第二列
        end = start
        while end <= start:
            end = random.randint(2, 6)
        
        # 第四列数据，从1到100的随机整数
        time = random.randint(0, 59)
        
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
    
    # 转换字典格式，计算每个time下相同start和end的组合的个数
    time_count_dict = {}
    for time, data_list in time_data.items():
        # 使用defaultdict来自动初始化计数器
        count_dict = defaultdict(int)
        for data in data_list:
            count_dict[data] += 1
        # 将每个time对应的数据和计数存入time_count_dict
        time_count_dict[time] = [(data[0], data[1], count) for data, count in count_dict.items()]

    return time_count_dict

# 打印结果
def print_results(time_count_dict):
    for time, entries in time_count_dict.items():
        for entry in entries:
            print(f"time: {time}, start: {entry[0]}, end: {entry[1]}, count: {entry[2]}")

    # 调用函数处理CSV文件并打印结果
    time_count_dict = process_csv()
    print_results(time_count_dict)
    for key in time_count_dict:
        print(key)
        print(time_count_dict[key])

pass_od = process_csv()

a = pass_od[1]

generate_data()
