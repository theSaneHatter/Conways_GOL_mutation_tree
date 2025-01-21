import numpy as np
import time
import cupy as cp
import timeit
 
arr = cp.array([1,2,3])


def GOL_clock(GOL_arr=np.random.randint(0,2,(15,15))):
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



# Create a 5x5 board with some initial cells
board = np.array([
    [1,0,0,1,0,1,0],
    [1,0,0,1,0  ,1,0],
    [1,0,0,1,0,1,0],
    [1,0,0,1,0,1,0],
    [1,0,0,1,0,1,0]
])

tim = timeit.timeit(GOL_clock,number=1000000)
print(tim,'seconds to run') #like  44 sec with 1mil 15x15