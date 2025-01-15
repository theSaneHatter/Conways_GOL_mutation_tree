import time
import cupy as cp
import numpy as np
import timeit



def GOL_clock(GOL_arr):
    rows, colms = GOL_arr.shape
    board = np.zeros((rows+2,colms+2),dtype=int)
    board[1:rows+1,1:colms+1] = GOL_arr[:,:]
    board_save = board.copy()

    shift_up = np.roll(board_save,1,0)
    board = board + shift_up
    
    shift_down = np.roll(board_save,-1,0)
    board = board + shift_down
    
    shift_right = np.roll(board_save,1,1)
    board = board + shift_right

    shift_left = np.roll(board_save,-1,1)
    board = board + shift_left

    
    
    board = np.where((board == 2) & (board == 3), 0,board)

    print(board)

arr = np.random.randint(0,2,(5,5))
board = [
    
    [0,1,0],
    [0,1,0],
    [0,1,0]
]
board = np.array(board)

print('Original Arr\n',board)
GOL_clock(board)