import numpy as np

initial_board =  np.array([
    [1,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,0,8,0,0,0,0,6,0],
    [2,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,0,0,0,0,0],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,0],
    [0,0,0,0,0,0,0,0,0]
])

knights_paths = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 8), (0, 7), (0, 6), (1, 6)],
    [(8, 0), (8, 1), (8, 2), (7, 2)],
    [(8, 8), (7, 8), (6, 8), (6, 7)],
]

knights_total = 19


def solve_sudoku_with_paths(grid, paths, target_sum=19):
    solutions = []
    board = np.array(grid, dtype=int)

    path_coords = [np.array(p) for p in paths]

    def is_valid(r, c, val):
        if val in board[r, :] or val in board[:, c]:
            return False

        box_r, box_c = (r // 3) * 3, (c // 3) * 3
        if val in board[box_r:box_r + 3, box_c:box_c + 3]:
            return False
        return True

    def check_paths_partial_or_full(complete=False):
        for p in path_coords:
            vals = board[p[:, 0], p[:, 1]]
            current_sum = np.sum(vals)
            has_empty = np.any(vals == 0)

            if complete:
                if current_sum < target_sum:
                    return False
            else:
                if not has_empty and current_sum < target_sum:
                    return False

        return True

    def backtrack(r=0, c=0):
        if r == 9:
            if check_paths_partial_or_full(complete=True):
                solutions.append(board.copy())
            return

        next_r, next_c = (r, c + 1) if c < 8 else (r + 1, 0)

        if board[r, c] != 0:
            if check_paths_partial_or_full(complete=False):
                backtrack(next_r, next_c)
        else:
            for val in range(1, 10):
                if is_valid(r, c, val):
                    board[r, c] = val
                    if check_paths_partial_or_full(complete=False):
                        backtrack(next_r, next_c)
                    board[r, c] = 0

    backtrack()
    return solutions

if __name__ == '__main__':
    all_solutions = solve_sudoku_with_paths(initial_board, knights_paths, target_sum=knights_total)
    print(f"Total solutions found: {len(all_solutions)}")
    for s in all_solutions:
        text = '\n'.join(''.join(f'{cell}' for cell in row) for row in s)
        print(text + '\n')

#Count: 14
#MD5:   9a9d9d9da9f1be6a54b140db6da6c7fa