import numpy as np
import time

def game_of_life(board):
    if len(board.shape) == 1:
        # Reshape the 1D array into a 2D array
        dim = int(np.sqrt(len(board)))
        if dim * dim != len(board):
            raise ValueError("1D array length must be a perfect square")
        board = board.reshape((dim, dim))

    # Get the shape of the board
    rows, cols = board.shape

    # Create a larger board to accommodate the edges
    larger_board = np.zeros((rows+2, cols+2))
    
    # Copy the original board to the center of the larger board
    larger_board[1:-1, 1:-1] = board
    
    # Create a new board to store the next generation
    new_board = np.copy(board)
    
    # Iterate over each cell in the original board
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            # Count the number of live neighbors
            live_neighbors = np.sum(larger_board[i-1:i+2, j-1:j+2]) - larger_board[i, j]
            
            # Apply the rules of Conway's Game of Life
            if larger_board[i, j] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                new_board[i-1, j-1] = 0
            elif larger_board[i, j] == 0 and live_neighbors == 3:
                new_board[i-1, j-1] = 1
    
    return new_board


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

# Create a 5x5 board with some initial cells
board = np.array([
    [1],
    [1],
    [1],
    [1],
    [1]
])
# Print the initial board
print("Initial Board:")
print(board)

# Apply one generation of Conway's Game of Life
new_board = game_of_life(board)

# Print the resulting board
print("\nResulting Board:")
print(new_board)

while True:
    start = time.time()
    board = game_of_life(board)
    end = time.time() - start
    print(f'time to compute board={round(end,6)}')
    print('\n')
    print(board)
    time.sleep(.25)