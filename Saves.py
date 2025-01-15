
def game_of_life_cuda(board):
    if len(board.shape) == 1:
        dim = int(np.sqrt(len(board)))
        if dim * dim != len(board):
            raise ValueError("1D array length must be a perfect square")
        board = board.reshape((dim, dim))

    rows, cols = board.shape
    larger_board = cp.zeros((rows+2, cols+2), dtype=cp.int32)
    larger_board[1:-1, 1:-1] = cp.asarray(board)

    new_board = cp.copy(larger_board[1:-1, 1:-1])

    @cp.fuse(kernel_name='game_of_life_kernel')
    def game_of_life_kernel(larger_board, new_board):
        i, j = cp.grid(2)
        if 0 <= i < new_board.shape[0] and 0 <= j < new_board.shape[1]:
            live_neighbors = cp.sum(larger_board[i-1:i+2, j-1:j+2]) - larger_board[i+1, j+1]
            if larger_board[i+1, j+1] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                new_board[i, j] = 0
            elif larger_board[i+1, j+1] == 0 and live_neighbors == 3:
                new_board[i, j] = 1

    game_of_life_kernel((rows, cols), (larger_board, new_board))

    return cp.asnumpy(new_board)