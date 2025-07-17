import numpy as np
import pickle
import hashlib

# 게임 설정
BOARD_SIZE = 15
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Q-테이블 로드
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

def board_to_hash(board):
    return hashlib.md5(board.tobytes()).hexdigest()

def is_valid_move(board, row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0

def count_open_lines(board, row, col, player, length):
    open_lines = 0
    temp_place = board[row][col] == 0
    if temp_place:
        board[row][col] = player
    
    for dr, dc in DIRECTIONS:
        for direction in [1, -1]:
            count = 1
            r, c = row, col
            for i in range(1, length):
                r, c = row + dr * i * direction, col + dc * i * direction
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                    count += 1
                else:
                    break
            r, c = row + dr * length * direction, col + dc * length * direction
            is_open = 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 0
            if count == length and is_open:
                open_lines += 1
    
    if temp_place:
        board[row][col] = 0
    return open_lines

def is_forbidden_move(board, row, col, player):
    if player != 1:
        return False
    open_threes = count_open_lines(board, row, col, player, 3)
    open_fours = count_open_lines(board, row, col, player, 4)
    return open_threes >= 2 or open_fours >= 2

def check_winner(board, player):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != player:
                continue
            for dr, dc in DIRECTIONS:
                count = 1
                for i in range(1, 5):
                    r, c = row + dr * i, col + dc * i
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                        count += 1
                    else:
                        break
                if count == 5:
                    return True
    return False

def get_valid_moves(board, player):
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if is_valid_move(board, row, col) and not (player == 1 and is_forbidden_move(board, row, col, player)):
                moves.append((row, col))
    return moves

def choose_action(board, player):
    state = board_to_hash(board)
    valid_moves = get_valid_moves(board, player)
    if not valid_moves:
        return None
    
    if state in q_table:
        max_q = max(q_table[state].values(), default=0)
        best_moves = [move for move, q in q_table[state].items() if q == max_q]
        return random.choice(best_moves) if best_moves else random.choice(valid_moves)
    return random.choice(valid_moves)

def print_board(board):
    print("   ", end="")
    for col in range(BOARD_SIZE):
        print(f"{col:2}", end=" ")
    print()
    for row in range(BOARD_SIZE):
        print(f"{row:2} ", end="")
        for col in range(BOARD_SIZE):
            if board[row][col] == 1:
                print("X ", end=" ")
            elif board[row][col] == -1:
                print("O ", end=" ")
            else:
                print(". ", end=" ")
        print()

def play_game():
    print("오목 게임 시작!")
    player_choice = input("흑돌(X, 선수) 또는 백돌(O, 후수)을 선택하세요 (black/white): ").lower()
    while player_choice not in ['black', 'white']:
        player_choice = input("잘못된 입력입니다. black 또는 white를 입력하세요: ").lower()
    
    player = 1 if player_choice == 'black' else -1
    ai = -player
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    current_player = 1  # 흑돌이 항상 선수
    
    if player == -1:  # 백돌 선택 시 AI가 먼저
        action = choose_action(board, 1)
        if action:
            row, col = action
            board[row][col] = 1
            print(f"AI가 ({row}, {col})에 흑돌을 놓았습니다.")
            print_board(board)
    
    while True:
        if current_player == player:
            print_board(board)
            try:
                row = int(input("행(row)을 입력하세요 (0-14): "))
                col = int(input("열(col)을 입력하세요 (0-14): "))
                if not is_valid_move(board, row, col):
                    print("유효하지 않은 위치입니다!")
                    continue
                if player == 1 and is_forbidden_move(board, row, col, player):
                    print("33 또는 44 금지 규칙 위반! 다시 입력하세요.")
                    continue
                board[row][col] = player
            except ValueError:
                print("숫자를 입력하세요!")
                continue
        else:
            action = choose_action(board, ai)
            if not action:
                print("무승부!")
                break
            row, col = action
            board[row][col] = ai
            print(f"AI가 ({row}, {col})에 {'흑돌' if ai == 1 else '백돌'}을 놓았습니다.")
        
        if check_winner(board, current_player):
            print_board(board)
            print(f"{'플레이어' if current_player == player else 'AI'} 승리!")
            break
        
        current_player = -current_player

if __name__ == "__main__":
    play_game()