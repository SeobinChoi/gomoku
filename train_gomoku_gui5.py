import numpy as np
import random
import pickle
import hashlib
import pygame
import time
import os
from tqdm import tqdm

"""
Gomoku Q‑Learning with sequential move numbering on stones.
- draws stones and their move order (1,2,3, …) correctly
- uses moves_log exclusively for rendering so numbering never breaks
- colour of the number is opposite to the stone colour for readability
"""

# ────────────────────────────────────
# Configuration
# ────────────────────────────────────
BOARD_SIZE          = 15
CELL_SIZE           = 40
BOARD_WIDTH         = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT        = BOARD_SIZE * CELL_SIZE + 100   # extra space for stats text
MAX_MOVES           = BOARD_SIZE * BOARD_SIZE
DIRECTIONS          = [(0, 1), (1, 0), (1, 1), (1, -1)]
FPS                 = 3        # GUI refresh cap
RENDER_MOVE_EVERY   = 5        # render every n moves
COL_LABELS          = 'ABCDEFGHIJKLMNO'

# Colours
BLACK        = (  0,   0,   0)
WHITE        = (255, 255, 255)
BOARD_COLOR  = (200, 200, 100)
LINE_COLOR   = (  0,   0,   0)
TEXT_COLOR   = (  0,   0,   0)

# Q‑Learning
ALPHA   = 0.1     # learning‑rate
GAMMA   = 0.9     # discount
EPSILON = 0.3     # ε‑greedy exploration
EPISODES = 10_000
RENDER_EVERY      = 100
MAX_Q_TABLE_SIZE  = 500_000

# ────────────────────────────────────
# Globals
# ────────────────────────────────────
q_table: dict[str, dict[tuple[int,int], float]] = {}
screen: pygame.Surface | None = None
clock:  pygame.time.Clock | None = None

# ────────────────────────────────────
# GUI helpers
# ────────────────────────────────────

def setup_gui() -> None:
    global screen, clock
    pygame.init()
    screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("Gomoku Q‑Learning")
    clock = pygame.time.Clock()


def draw_board(board: np.ndarray, moves_log: list[tuple[int,int]] | None = None) -> None:
    """Render grid, stones, and sequential numbers."""
    assert screen is not None, "GUI not initialised"
    # background & grid
    screen.fill(BOARD_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR,
                         (CELL_SIZE, (i + .5) * CELL_SIZE),
                         (BOARD_WIDTH - CELL_SIZE, (i + .5) * CELL_SIZE))
        pygame.draw.line(screen, LINE_COLOR,
                         ((i + .5) * CELL_SIZE, CELL_SIZE),
                         ((i + .5) * CELL_SIZE, BOARD_HEIGHT - CELL_SIZE - 100))

    # render stones strictly in the order they were played so numbers are sequential
    if moves_log:
        font = pygame.font.Font(None, 24)
        for idx, (row, col) in enumerate(moves_log):
            player      = 1 if idx % 2 == 0 else -1
            stone_color = BLACK if player == 1 else WHITE
            num_color   = WHITE if player == 1 else BLACK
            centre      = ((col + .5) * CELL_SIZE, (row + .5) * CELL_SIZE)
            pygame.draw.circle(screen, stone_color, centre, CELL_SIZE // 2 - 2)
            text = font.render(str(idx + 1), True, num_color)
            text_rect = text.get_rect(center=centre)
            screen.blit(text, text_rect)

    pygame.display.flip()


def display_stats(episode: int, avg_reward: float, q_table_size: int,
                   move_count: int, render_time: float) -> None:
    font = pygame.font.Font(None, 24)
    text = font.render(
        f"Ep {episode}/{EPISODES} | AvgR {avg_reward:.2f} | Q {q_table_size} | Moves {move_count} | rt {render_time:.3f}s",
        True, TEXT_COLOR)
    screen.fill(BOARD_COLOR, (0, BOARD_HEIGHT - 100, BOARD_WIDTH, 100))
    screen.blit(text, (10, BOARD_HEIGHT - 80))
    pygame.display.flip()

# ────────────────────────────────────
# Game‑logic helpers
# ────────────────────────────────────

def board_to_hash(board: np.ndarray) -> str:
    return hashlib.md5(board.tobytes()).hexdigest()


def is_valid_move(board: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 0


def count_open_lines(board: np.ndarray, r: int, c: int, player: int, length: int) -> int:
    open_lines = 0
    board[r][c] = player  # temporarily place stone
    for dr, dc in DIRECTIONS:
        for sign in (1, -1):
            cnt = 1
            for i in range(1, length):
                nr, nc = r + dr * i * sign, c + dc * i * sign
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == player:
                    cnt += 1
                else:
                    break
            nr, nc = r + dr * length * sign, c + dc * length * sign
            if cnt == length and 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == 0:
                open_lines += 1
    board[r][c] = 0  # revert
    return open_lines


def is_forbidden_move(board: np.ndarray, r: int, c: int, player: int) -> bool:
    if player != 1:
        return False
    return (count_open_lines(board, r, c, player, 3) >= 2 or
            count_open_lines(board, r, c, player, 4) >= 2)


def check_winner(board: np.ndarray, player: int) -> bool:
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != player:
                continue
            for dr, dc in DIRECTIONS:
                if all(0 <= r + dr*i < BOARD_SIZE and 0 <= c + dc*i < BOARD_SIZE and
                       board[r + dr*i][c + dc*i] == player for i in range(5)):
                    return True
    return False


def get_valid_moves(board: np.ndarray, player: int):
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            if is_valid_move(board, r, c) and not (player == 1 and is_forbidden_move(board, r, c, player))]


def choose_action(board: np.ndarray, player: int, epsilon: float):
    state       = board_to_hash(board)
    valid_moves = get_valid_moves(board, player)
    if not valid_moves:
        return None
    if random.random() < epsilon:
        return random.choice(valid_moves)
    if state not in q_table:
        q_table[state] = {m: 0. for m in valid_moves}
    max_q   = max(q_table[state].values())
    best_mv = [m for m, q in q_table[state].items() if q == max_q]
    return random.choice(best_mv)


def trim_q_table():
    if len(q_table) <= MAX_Q_TABLE_SIZE:
        return
    for key in random.sample(list(q_table.keys()), len(q_table)//10):
        q_table.pop(key, None)


def save_game_log(episode: int, board: np.ndarray, moves_log: list[tuple[int,int]]):
    os.makedirs('logs', exist_ok=True)
    img_path = os.path.join('logs', f'ep_{episode}.png')
    pygame.image.save(screen, img_path)
    txt_path = os.path.join('logs', 'log.txt')
    with open(txt_path, 'a') as f:
        f.write(f"\nEpisode {episode} | image: {img_path}\n")
        for i, (r, c) in enumerate(moves_log):
            f.write(f"{i+1}. {'B' if i%2==0 else 'W'} {COL_LABELS[c]}{r}\n")

# ────────────────────────────────────
# Training loop
# ────────────────────────────────────

def train():
    setup_gui()
    total_rewards: list[float] = []
    total_rt = 0.
    moves_log: list[tuple[int,int]] = []
    running = True

    for ep in tqdm(range(1, EPISODES+1), unit="ep"):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), int)
        moves_log.clear()
        player      = 1   # Black first
        reward_sum  = 0.
        move_cnt    = 0
        draw_board(board, moves_log)
        clock.tick(FPS)

        while move_cnt < MAX_MOVES and running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
            if not running:
                break

            start = time.time()
            action = choose_action(board, player, EPSILON)
            if action is None:
                tqdm.write(f"Ep {ep}: draw, {move_cnt} moves")
                save_game_log(ep, board, moves_log)
                break

            r, c = action
            board[r][c] = player
            moves_log.append((r, c))
            move_cnt += 1

            if move_cnt % RENDER_MOVE_EVERY == 0 or check_winner(board, player):
                draw_board(board, moves_log)
                total_rt += time.time() - start
                clock.tick(FPS)

            if check_winner(board, player):
                rew = 100
                reward_sum += rew
                state = board_to_hash(board)
                q_table.setdefault(state, {(r, c): 0})[(r, c)] = (1-ALPHA)*q_table[state].get((r,c),0)+ALPHA*rew
                tqdm.write(f"Ep {ep}: {'Black' if player==1 else 'White'} wins in {move_cnt}")
                save_game_log(ep, board, moves_log)
                break

            # Q‑update
            state = board_to_hash(board)
            q_table.setdefault(state, {(r, c): 0})
            next_moves = get_valid_moves(board, -player)
            next_max_q = 0 if not next_moves else max(q_table.setdefault(board_to_hash(board), {m:0 for m in next_moves}).values())
            q_table[state][(r, c)] = (1-ALPHA)*q_table[state][(r, c)] + ALPHA*(0 + GAMMA*next_max_q)

            player *= -1

            if ep % RENDER_EVERY == 0:
                avg_r = sum(total_rewards[-1000:]) / max(1, len(total_rewards[-1000:]))
                display_stats(ep, avg_r, len(q_table), move_cnt, total_rt/max(1, move_cnt))

        total_rewards.append(reward_sum)
        trim_q_table()

        if ep % 1000 == 0:
            avg_r = sum(total_rewards[-1000:]) / max(1, len(total_rewards[-1000:]))
            tqdm.write(f"Ep {ep}: avgR {avg_r:.2f} | Q {len(q_table)} | moves {move_cnt}")
            total_rt = 0.

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)
    pygame.quit()

# ────────────────────────────────────

if __name__ == "__main__":
    train()
