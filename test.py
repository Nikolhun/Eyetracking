import numpy as np

def find_closest_in_array(array, value):
    '''
    Finds closest number and its coordinates in array
    :param array: numpy array with at least two rows
    :param value: value, that is rearched
    :return: [number, row, column]
    '''
    ro = -1  # not 0 because len array returns from 1 not from 0
    closest_from_every_row = []
    closest_from_every_row_row = []
    closest_from_every_row_column = []
    for i in range(0, len(array)):
        ro = ro + 1
        array_line = np.asarray(array[i])
        idx = (np.abs(array_line - value)).argmin()  # get position
        closest_from_every_row.append(array_line[idx])  # closest numbers from row
        closest_from_every_row_row.append(ro)  # coordinate row
        closest_from_every_row_column.append(idx)  # coordinate column

    closest_from_every_row = np.asarray(closest_from_every_row)
    idx_from_every_row = (np.abs(closest_from_every_row - value)).argmin()  # get position prom closest numbers in row

    return closest_from_every_row[idx_from_every_row], closest_from_every_row_row[idx_from_every_row],\
           closest_from_every_row_column[idx_from_every_row]


array = np.zeros((20, 20), np.uint8)
array[1, 1] = 100
array[2, 1] = 50
array[10, 5] = 54
array[19, 0] = 4
array[19, 4] = 4
array[5, 10] = 100

number, row, column = find_closest_in_array(array, 51)
print(number, row, column)

