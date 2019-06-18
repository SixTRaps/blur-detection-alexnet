import os
import re
import math

predict_file_addr = 'predict_prob.txt'
new_file_addr = 'average_accuracy.txt'

with open(new_file_addr,'w') as f2:
    with open(predict_file_addr, 'r') as f1:
        data_list =f1.readlines()
        totalLine = len(data_list)
        transform_list = []
        for i in range(totalLine):
            transform_list.append(data_list[i].split(' '))
        #print(transform_list)
        count = 1
        sum = 0.0
        for i in range(totalLine):
            if i == totalLine-1:
                pos = transform_list[i][0].index('_',transform_list[i][0].index('_')+1)
                previous_pos = transform_list[i-1][0].index('_',transform_list[i-1][0].index('_')+1)
                if transform_list[i][0][0:pos] == transform_list[i-1][0][0:previous_pos]:
                    break
                else:
                    f2.write(transform_list[i][0][0:pos]+'.jpeg' + ' ' + transform_list[i][1] + ' ' + transform_list[i][2])
                    break
            comp1 = re.search(re.compile(r"[0-9]+"),data_list[i]).group(0)
            comp2 = re.search(re.compile(r"[0-9]+"),data_list[i+1]).group(0)
            if comp2 == comp1:
                if sum == 0.0:
                    sum+=float(transform_list[i][1])
                sum += float(transform_list[i+1][1])
                count+=1

            else:
                if sum == 0.0:
                    sum+=float(transform_list[i][1])
                average_accuracy = sum / count

                count = 1
                sum = 0.0
                pos = transform_list[i][0].index('_',transform_list[i][0].index('_')+1)
                f2.write(transform_list[i][0][0:pos]+'.jpeg' + ' ' + str(average_accuracy) + ' ' + transform_list[i][2])

f1.close()
f2.close()


print("Finished.")
