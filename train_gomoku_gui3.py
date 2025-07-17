import numpy as np
import random
import pickle
import hashlib
import pygame
import time
import os
from tqdm import tqdm

# 게임 설정
BOARD_SIZE = 15
CELL_SIZE = 40
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE + 100  # 텍스트 표시 공간
MAX_MOVES = BOARD_SIZE * BOARD_SIZE
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
FPS = 3  # GUI 렌더링 속도 제어
RENDER_MOVE_EVERY = 5  # 5 이동마다 렌더링
COL_LABELS = 'ABCDEFGHIJKLMNO'  # 열 A~O

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (200, 200, 100)
LINE_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)

# Q-Learning 파라미터
ALPHA = 0.1  # 학습률
GAMMA = 0.9  # 할인율
EPSILON = 0.3  # 탐색 확률
EPISODES = 10000  # 학습 에포크 수
RENDER_EVERY = 100  # 상태 텍스트 업데이트 주기
MAX_Q_TABLE_SIZE = 500000  # Q-테이블 크기 제한

# 글로벌 변수
q_table = {}
screen = None
clock = None
log_file = None  # 로그 파일 핸들 초기화

def setup_gui():
    global screen, clock
    pygame.init()
    screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("Gomoku Training")
    clock = pygame.time.Clock()
    screen.fill(BOARD_COLOR)
    draw_board(np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int))

def draw_board(board):
    screen.fill(BOARD_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE, (i + 0.5) * CELL_SIZE), (BOARD_WIDTH - CELL_SIZE, (i + 0.5) * CELL_SIZE))
        pygame.draw.line(screen, LINE_COLOR, ((i + 0.5) * CELL_SIZE, CELL_SIZE), ((i + 0.5) * CELL_SIZE, BOARD_HEIGHT - CELL_SIZE - 100))
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != 0:
                color = BLACK if board[row][col] == 1 else WHITE
                pygame.draw.circle(screen, color, ((col + 0.5) * CELL_SIZE, (row + 0.5) * CELL_SIZE), CELL_SIZE // 2 - 2)
    pygame.display.flip()

def display_stats(episode, avg_reward, q_table_size, move_count, render_time):
    font = pygame.font.Font(None, 24)
    text = font.render(f"Episode: {episode}/{EPISODES}  Avg Reward: {avg_reward:.2f}  Q-Table Size: {q_table_size}  Moves: {move_count}  Render Time: {render_time:.3f}s", True, TEXT_COLOR)
    screen.fill(BOARD_COLOR, (0, BOARD_HEIGHT - 100, BOARD_WIDTH, 100))
    screen.blit(text, (10, BOARD_HEIGHT - 80))
    pygame.display.flip()

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

def choose_action(board, player, epsilon):
    state = board_to_hash(board)
    valid_moves = get_valid_moves(board, player)
    if not valid_moves:
        return None
    
    if random.random() < epsilon:
        return random.choice(valid_moves)
    
    if state not in q_table:
        q_table[state] = {move: 0 for move in valid_moves}
    
    max_q = max(q_table[state].values(), default=0)
    best_moves = [move for move, q in q_table[state].items() if q == max_q]
    return random.choice(best_moves) if best_moves else random.choice(valid_moves)

def trim_q_table():
    if len(q_table) > MAX_Q_TABLE_SIZE:
        states = list(q_table.keys())
        random.shuffle(states)
        for state in states[:len(states) // 10]:  # 10% 제거
            del q_table[state]

def get_coord_string(row, col):
    """숫자와 알파벳으로 좌표 변환 (예: (0, 0) -> A0, (1, 2) -> B2)"""
    return f"{COL_LABELS[col]}{row}"

def save_game_log(episode, board, moves_log):
    """기보 이미지와 로그를 logs 폴더에 저장"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # logs 폴더 생성
    image_filename = os.path.join(log_dir, f"episode_{episode}_board.png")
    try:
        pygame.display.flip()  # 화면 업데이트 보장
        pygame.image.save(screen, image_filename)
    except Exception as e:
        print(f"Failed to save image {image_filename}: {str(e)}")
        return  # 저장 실패 시 로그 기록 건너뜀
    log_filename = os.path.join(log_dir, "log.txt")
    with open(log_filename, 'a') as log_file:
        log_entry = f"\nEpisode {episode}:\nImage: {image_filename}\nMoves:\n"
        for i, (row, col) in enumerate(moves_log):
            player = "Black" if i % 2 == 0 else "White"
            log_entry += f"{i + 1}. {player} at {get_coord_string(row, col)}\n"
        log_file.write(log_entry)

def train():
    global log_file
    setup_gui()
    total_rewards = []
    total_render_time = 0
    running = True
    moves_log = []  # 각 에포크의 이동 기록
    
    for episode in tqdm(range(EPISODES), desc="Training", unit="episode"):
        if not running:
            break
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        current_player = 1
        episode_reward = 0
        move_count = 0
        moves_log.clear()  # 에포크마다 이동 기록 초기화
        
        draw_board(board)  # 초기 보드 렌der링
        clock.tick(FPS)
        
        while move_count < MAX_MOVES:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break
            
            start_time = time.time()
            action = choose_action(board, current_player, EPSILON)
            if not action:
                tqdm.write(f"Episode {episode + 1}: Draw after {move_count} moves")
                save_game_log(episode + 1, board, moves_log)
                draw_board(board)  # 무승부 시 최종 보드 렌더링
                break
            
            row, col = action
            board[row][col] = current_player
            moves_log.append((row, col))  # 이동 기록
            if move_count % RENDER_MOVE_EVERY == 0 or check_winner(board, current_player):
                draw_board(board)  # 5 이동마다 또는 승리 시 렌더링
                render_time = time.time() - start_time
                total_render_time += render_time
                clock.tick(FPS)
            
            if check_winner(board, current_player):
                reward = 100
                episode_reward += reward
                state = board_to_hash(board)
                if state not in q_table:
                    q_table[state] = {(row, col): 0}
                q_table[state][(row, col)] = (1 - ALPHA) * q_table[state][(row, col)] + ALPHA * reward
                tqdm.write(f"Episode {episode + 1}: {['White', 'Black'][current_player == 1]} wins after {move_count + 1} moves")
                save_game_log(episode + 1, board, moves_log)
                draw_board(board)  # 승리 시 최종 보드 렌더링
                break
            elif current_player == 1 and is_forbidden_move(board, row, col, current_player):
                reward = -50
                tqdm.write(f"Episode {episode + 1}: Forbidden move by Black after {move_count + 1} moves")
                save_game_log(episode + 1, board, moves_log)
                draw_board(board)  # 금지된 수 시 최종 보드 렌더링
                break
            else:
                reward = 0
            
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
            
            if episode % RENDER_EVERY == 0:
                avg_reward = sum(total_rewards[-1000:]) / min(len(total_rewards), 1000) if total_rewards else 0
                display_stats(episode + 1, avg_reward, len(q_table), move_count, total_render_time / max(1, move_count))
        
        total_rewards.append(episode_reward)
        trim_q_table()
        
        if (episode + 1) % 1000 == 0:
            avg_reward = sum(total_rewards[-1000:]) / min(len(total_rewards), 1000)
            avg_moves = move_count  # 최근 에포크의 이동 수
            tqdm.write(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Q-Table Size = {len(q_table)}, Moves = {avg_moves}, Avg Render Time = {total_render_time / max(1, move_count):.3f}s")
            total_render_time = 0
    
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)
    pygame.quit()

if __name__ == "__main__":
    train()