import os


CONCATPATH = 'average_accuracy.txt'
positive = 0.0
negative = 0.0
true_positive = 0.0
true_negative = 0.0

with open(CONCATPATH, 'r') as f:
    all_data_list = f.readlines()
    transform_list = []
    for i in range(len(all_data_list)):
        transform_list.append(all_data_list[i].split(' '))

    for i in range(len(all_data_list)):
        if transform_list[i][2].rstrip('\n') == 'clear':
            positive+=1
        elif transform_list[i][2].rstrip('\n') == 'blur':
            negative+=1

        if (transform_list[i][2].rstrip('\n') == 'clear' and
            float(transform_list[i][1])>=0.5):
            true_positive+=1
        elif (transform_list[i][2].rstrip('\n') == 'blur' and
            float(transform_list[i][1])<0.5):
            true_negative+=1

accuracy = (true_positive+true_negative) / (positive+negative)
recall = true_positive / positive
print("Accuracy: %f"%accuracy)
print("Recall: %f"%recall)
