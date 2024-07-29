import pandas as pd
import os

import csv

# 定义时间间隔和数量和的字典
interval_sums = {}

# 遍历每个CSV文件
for csv_file in ["predict\start_station_0.csv", "predict\start_station_1.csv", "predict\start_station_2.csv", "predict\start_station_3.csv", "predict\start_station_4.csv"]:
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行

        for row in reader:
            time = int(row[0])
            count = int(row[1])

            # 计算对应时间间隔的数量和
            interval = (time // 10) * 10
            if csv_file not in interval_sums:
                interval_sums[csv_file] = {}
            if interval in interval_sums[csv_file]:
                interval_sums[csv_file][interval] += count
            else:
                interval_sums[csv_file][interval] = count

# 获取所有时间间隔
intervals = sorted(set(interval for sums in interval_sums.values() for interval in sums.keys()))

# 将字典转换为列表
result = [[interval_sums[csv_file].get(interval, 0) for interval in intervals] for csv_file in interval_sums.keys()]

# 保存结果到新的CSV文件
with open("toy_start_result.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time"] + ["Sum{}".format(i+1) for i in range(5)])  # 写入标题行
    writer.writerows(zip(intervals, *result))  # 写入时间和各个CSV文件的数量和