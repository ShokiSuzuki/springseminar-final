import csv


PATH  = 'test_result.csv'
PATH1 = 'test_result_cifar100.csv'
PATH2 = 'test_result_emnist.csv'


with open(PATH1) as f1, open(PATH2) as f2, open(PATH, 'w') as f:
    s1 = f1.read()
    s2 = f2.read()
    s  = s1 + s2[20:]
    f.write(s)

