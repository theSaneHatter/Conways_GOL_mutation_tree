import time
import numpy as np
import timeit


# returns none if array is all 0s
def GOL_clock(GOL_arr):
    if np.sum(GOL_arr) <= 0:
        print('\033[32mWorning from GOL_clock(): Entered array are all zeros\033[0m')
        return None
    rows, colms = GOL_arr.shape
    board_save = np.zeros((rows+2,colms+2),dtype=int)
    board_save[1:rows+1,1:colms+1] = GOL_arr[:,:]
    

    shift_up = np.roll(board_save,1,0)
    shift_down = np.roll(board_save,-1,0)
    shift_right = np.roll(board_save,1,1)
    shift_left = np.roll(board_save,-1,1)

    shift_RU = np.roll(shift_right,1,1)
    shift_RD = np.roll(shift_right,1,-1)
    shift_LU = np.roll(shift_left,1,1)
    shift_LD = np.roll(shift_left,1,-1)
    
    neighbers = shift_up+shift_down+shift_right+shift_left+shift_RU+shift_RD+shift_LU+shift_LD

    rectified_board = neighbers.copy()

    rectified_board = np.where((rectified_board >= 2) & (rectified_board <= 3), 1, 0)


    
    return rectified_board


def GOL_clock2():
    GOL_arr = np.random.randint(0,2,(15,15))

    if np.sum(GOL_arr) <= 0:
        print('\033[32mWorning from GOL_clock(): Entered array are all zeros\033[0m')
        return None
    rows, colms = GOL_arr.shape
    board_save = np.zeros((rows+2,colms+2),dtype=int)
    board_save[1:rows+1,1:colms+1] = GOL_arr[:,:]
    

    shift_up = np.roll(board_save,1,0)
    shift_down = np.roll(board_save,-1,0)
    shift_right = np.roll(board_save,1,1)
    shift_left = np.roll(board_save,-1,1)

    shift_RU = np.roll(shift_right,1,0)
    shift_RD = np.roll(shift_right,-1,0)
    shift_LU = np.roll(shift_left,1,0)
    shift_LD = np.roll(shift_left,-1,0)
    
    neighbers = shift_up+shift_down+shift_right+shift_left+shift_RU+shift_RD+shift_LU+shift_LD

    rectified_board = np.where((board_save == 1) & ((neighbers < 2) | (neighbers > 3)), 0, board_save)
    rectified_board = np.where((board_save == 0) & (neighbers == 3), 1, rectified_board)
    
    return rectified_board


arr = np.random.randint(0,2,(5,5))
board = [
    
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [1,1,1]
]
board = np.array(board)

print('Original Arr\n',board)
# print('GOL_CLOCK2(board):\n',GOL_clock2(board))

t = time_taken = timeit.timeit(GOL_clock2, number=10000)  # Run 1000 times
print(t)
