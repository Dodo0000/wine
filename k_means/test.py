from kmeans import kmeans

obj = kmeans('./data.csv')
result = obj.get_result(3)
# print(result[0])
# print(result[1])
print(result[2])
