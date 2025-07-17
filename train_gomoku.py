import numpy as np
import random
import pickle
import hashlib
from tqdm import tqdm

# 게임 설정
BOARD_SIZE = 15
MAX_MOVES = BOARD_SIZE * BOARD_SIZE
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Q-Learning 파라미터
ALPHA = 0.1  # 학습률
GAMMA = 0.9  # 할인율
EPSILON = 0.1  # 탐색 확률
EPISODES = 10000  # 학습 에포크 수

# Q-테이블 (상태-행동 쌍)
q_table = {}

def board_to_hash(board):
    """보드 상태를 해시로 변환하여 메모리 절약"""
    return hashlib.md5(board.tobytes()).hexdigest()

def is_valid_move(board, row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0

def count_open_lines(board, row, col, player, length):
    """열린 n연속(3 또는 4)의 개수 계산"""
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
    """흑돌의 33, 44 금지 규칙 체크"""
    if player != 1:
        return False
    open_threes = count_open_lines(board, row, col, player, 3)
    open_fours = count_open_lines(board, row, col, player, 4)
    return open_threes >= 2 or open_fours >= 2

def check_winner(board, player):
    """승리 조건 체크"""
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
    """유효한 행동 목록 반환"""
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if is_valid_move(board, row, col) and not (player == 1 and is_forbidden_move(board, row, col, player)):
                moves.append((row, col))
    return moves

def choose_action(board, player, epsilon):
    """ε-greedy로 행동 선택"""
    state = board_to_hash(board)
    valid_moves = get_valid_moves(board, player)
    if not valid_moves:
        return None
    
    if random.random() < epsilon:
        return random.choice(valid_moves)
    
    if state not in q_table:
        q_table[state] = {move: 0 for move in valid_moves}
    
    # 최대 Q값을 가진 행동 선택
    max_q = max(q_table[state].values(), default=0)
    best_moves = [move for move, q in q_table[state].items() if q == max_q]
    return random.choice(best_moves) if best_moves else random.choice(valid_moves)

def train():
    total_rewards = []
    
    for episode in tqdm(range(EPISODES), desc="Training", unit="episode"):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        current_player = 1  # 흑돌 시작
        episode_reward = 0
        move_count = 0
        
        while move_count < MAX_MOVES:
            action = choose_action(board, current_player, EPSILON)
            if not action:
                break  # 더 이상 유효한 수 없음
            
            row, col = action
            board[row][col] = current_player
            
            # 보상 계산
            if check_winner(board, current_player):
                reward = 100
                episode_reward += reward
                state = board_to_hash(board)
                if state not in q_table:
                    q_table[state] = {(row, col): 0}
                q_table[state][(row, col)] = (1 - ALPHA) * q_table[state][(row, col)] + ALPHA * reward
                break
            elif current_player == 1 and is_forbidden_move(board, row, col, current_player):
                reward = -50
            else:
                reward = 0
            
            # Q-테이블 업데이트
            state = board_to_hash(board)
            if state not in q_table:
                q_table[state] = {(row, col): 0}
            
            next_valid_moves = get_valid_moves(board, -current_player)
            if next_valid_moves:
                next_state = board_to_hash(board)
                if next_state not in q_table:
                    q_table[next_state] = {(r, c): 0 for r, c in next_valid_moves}
                next_max_q = max(q_table[next_state].values(), default=0)
            else:
                next_max_q = 0
            
            q_table[state][(row, col)] = (1 - ALPHA) * q_table[state][(row, col)] + ALPHA * (reward + GAMMA * next_max_q)
            
            episode_reward += reward
            current_player = -current_player
            move_count += 1
        
        total_rewards.append(episode_reward)
        
        # 진행 상황 출력
        if (episode + 1) % 1000 == 0:
            avg_reward = sum(total_rewards[-1000:]) / min(len(total_rewards), 1000)
            tqdm.write(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Q-Table Size = {len(q_table)}")
    
    # Q-테이블 저장
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

if __name__ == "__main__":
    train()