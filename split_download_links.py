import math

input_file = 'download_links.txt'
num_splits = 10

# 读取所有URL
with open(input_file, 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

total = len(urls)
per_file = math.ceil(total / num_splits)

for i in range(num_splits):
    start = i * per_file
    end = min((i + 1) * per_file, total)
    with open(f'filtered_urls_{i+1}.txt', 'w') as fout:
        fout.write('\n'.join(urls[start:end])) 