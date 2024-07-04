import csv
from collections import defaultdict
import random
import csv



# 定义生成数据的函数
def generate_data():
    # 准备数据列表
    data_list = []
    
    # 生成数据
    for i in range(0, 200):
        # 第一列数据
        passenger = i
        
        # 第二列数据，从1到5的随机整数
        start = random.randint(0, 4)
        
        # 第三列数据，从2到6的随机整数，且大于第二列
        end = start
        while end <= start:
            end = random.randint(1, 5)
        
        # 第四列数据，从1到100的随机整数
        time = random.randint(0, 60)
        
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

print(33//9)