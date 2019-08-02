

import numpy as np 

#import tensorflow as tf

first_list = [3,4,5,6,7,8,9]
temp = [[1,2], first_list]
for list in temp:
    while(list):
        print(list.pop())
    
print(len(first_list))
print(first_list.pop())


#[batch_size, length, width, channels]
output = np.zeros([3,3,3,5])

print(output.shape)
#print(output)

for batch in range(0, 3):
    for length in range(0, 3):
        for width in range(0, 3):
            for channel in range(0, 5):
                l = (3*batch) + length
                w = (3*l) + width
                c = (3*w) + channel
                if(channel%2 > 0):
                    output[batch][length][width][channel] = -c 
                else:
                    output[batch][length][width][channel] = c 

print(output.shape)
#print(output)

non_channel_axis = tuple(range(0, len(output.shape)-1))
print(non_channel_axis) 

max = np.amax(np.abs(output), axis = non_channel_axis, keepdims=True)
print(max.shape)
print(max)

normalized = output / max
print(normalized.shape)
#print(normalized)

# I should test how to get the differences
# 
# I need to get a 3x3x3x1 array for each of the 3 channels 
split = np.split(normalized, normalized.shape[-1], axis=-1)
#print(split[0].shape)
check_for_inverse_corr = True
correlation_grid = []

for output in split:
        # we simply use subtract to get the differences, and abs value to get the magnitude
        pos_diff = np.abs(normalized - output)
        # we sum together for all axis except the last one
        pos_diff_sum = np.sum(pos_diff, axis = non_channel_axis)
        # if we aren't doing inverse checks, then we're done
        min_diff = pos_diff_sum
            
        # If we want to also check for inverse (negative) correlations 
        if (check_for_inverse_corr):
            # do the exact sime but with the inverse of the split
            neg = -output
            neg_diff = np.abs(normalized - neg)
            neg_diff_sum = np.sum(neg_diff, axis = non_channel_axis)
            min_diff = np.minimum(pos_diff_sum, neg_diff_sum)

    
        correlation_grid.append(min_diff)

correlation_grid = np.array(correlation_grid)
print(correlation_grid)

arg_sorted = np.argsort(correlation_grid, axis=None)
print(arg_sorted.shape)
print(arg_sorted)
sorted = np.unravel_index(arg_sorted, correlation_grid.shape) 
print(sorted)
print(correlation_grid[sorted])

working_correlation_grid = correlation_grid.copy()
working_correlation_grid[1] = np.full((working_correlation_grid.shape[0],), np.nan)
working_correlation_grid[:, 1] = np.nan
print(working_correlation_grid)



print("testing")
test = np.stack(sorted, axis=-1)
print(test.shape)
print(test)
test = np.concatenate((sorted[0], sorted[1].T), axis=-1)
print(test.shape)
print(test)


values = []
index_tuples = []
for index in range(0,len(sorted[0])):
    if(sorted[0][index] != sorted[1][index]):
        index_tuples.append((sorted[0][index], sorted[1][index]))
        value = correlation_grid[sorted[0][index]][sorted[1][index]]
        if (value > 0.0):
            values.append(value)

index_tuples = np.array(index_tuples)
print(index_tuples.shape)
print(index_tuples)

values= np.array(values)
print(values.shape)
print(values)


print("done")