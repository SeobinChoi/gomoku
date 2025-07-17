import pygame
import asyncio
import platform
import math

# 게임 설정
BOARD_SIZE = 15
CELL_SIZE = 40
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE
FPS = 60

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (200, 200, 100)
LINE_COLOR = (0, 0, 0)
BUTTON_COLOR = (150, 150, 150)
TEXT_COLOR = (0, 0, 0)

# 게임 상태
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]  # 0: 빈칸, 1: 흑돌, -1: 백돌
screen = None
current_player = None  # 초기 None, 선택 후 설정
player_choice = None   # 플레이어가 흑(1) 또는 백(-1)
game_started = False    # 게임 시작 여부

def setup():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("Gomoku")
    draw_selection_screen()

def draw_selection_screen():
    screen.fill(BOARD_COLOR)
    font = pygame.font.Font(None, 36)
    # 흑돌 버튼
    black_button = pygame.Rect(BOARD_WIDTH//4, BOARD_HEIGHT//2 - 50, 150, 50)
    pygame.draw.rect(screen, BUTTON_COLOR, black_button)
    text = font.render("Black (First)", True, TEXT_COLOR)
    screen.blit(text, (black_button.x + 10, black_button.y + 10))
    # 백돌 버튼
    white_button = pygame.Rect(BOARD_WIDTH//2 + 50, BOARD_HEIGHT//2 - 50, 150, 50)
    pygame.draw.rect(screen, BUTTON_COLOR, white_button)
    text = font.render("White (Second)", True, TEXT_COLOR)
    screen.blit(text, (white_button.x + 10, white_button.y + 10))
    pygame.display.flip()

def draw_board():
    screen.fill(BOARD_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE, (i + 0.5) * CELL_SIZE), (BOARD_WIDTH - CELL_SIZE, (i + 0.5) * CELL_SIZE))
        pygame.draw.line(screen, LINE_COLOR, ((i + 0.5) * CELL_SIZE, CELL_SIZE), ((i + 0.5) * CELL_SIZE, BOARD_HEIGHT - CELL_SIZE))
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != 0:
                draw_stone(row, col, board[row][col])
    pygame.display.flip()

def draw_stone(row, col, player):
    color = BLACK if player == 1 else WHITE
    pygame.draw.circle(screen, color, ((col + 0.5) * CELL_SIZE, (row + 0.5) * CELL_SIZE), CELL_SIZE // 2 - 2)

def is_valid_move(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0

def count_open_lines(row, col, player, length):
    """특정 위치에 돌을 놓았을 때 열린 n연속(3 또는 4)의 개수를 계산"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    open_lines = 0
    temp_place = board[row][col] == 0
    if temp_place:
        board[row][col] = player
    
    for dr, dc in directions:
        for direction in [1, -1]:
            count = 1
            r, c = row, col
            # 연속된 돌 세기
            for i in range(1, length):
                r, c = row + dr * i * direction, col + dc * i * direction
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                    count += 1
                else:
                    break
            # 열린 끝 확인
            r, c = row + dr * length * direction, col + dc * length * direction
            is_open = 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 0
            if count == length and is_open:
                open_lines += 1
    
    if temp_place:
        board[row][col] = 0
    return open_lines

def is_forbidden_move(row, col, player):
    """흑돌의 33, 44 금지 규칙 체크"""
    if player != 1:  # 흑돌에만 적용
        return False
    open_threes = count_open_lines(row, col, player, 3)
    open_fours = count_open_lines(row, col, player, 4)
    return open_threes >= 2 or open_fours >= 2

def place_stone(row, col, player):
    if is_valid_move(row, col) and not (player == 1 and is_forbidden_move(row, col, player)):
        board[row][col] = player
        draw_stone(row, col, player)
        pygame.display.flip()
        return True
    return False

def check_winner(player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != player:
                continue
            for dr, dc in directions:
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

def evaluate_board():
    score = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 0:
                continue
            player = board[row][col]
            for dr, dc in directions:
                count = 0
                for i in range(-4, 5):
                    r, c = row + dr * i, col + dc * i
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                        count += 1
                if count >= 5:
                    score += 100000 * player
                elif count == 4:
                    score += 1000 * player
                elif count == 3:
                    score += 100 * player
                elif count == 2:
                    score += 10 * player
    return score

def minimax(depth, alpha, beta, maximizing_player):
    if depth == 0 or check_winner(1) or check_winner(-1):
        return evaluate_board(), None

    best_move = None
    player = -1 if maximizing_player else 1
    if maximizing_player:
        max_eval = -math.inf
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if is_valid_move(row, col) and not (player == 1 and is_forbidden_move(row, col, player)):
                    board[row][col] = player
                    eval_score, _ = minimax(depth - 1, alpha, beta, False)
                    board[row][col] = 0
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = (row, col)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if is_valid_move(row, col) and not (player == 1 and is_forbidden_move(row, col, player)):
                    board[row][col] = player
                    eval_score, _ = minimax(depth - 1, alpha, beta, True)
                    board[row][col] = 0
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = (row, col)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
        return min_eval, best_move

def ai_move():
    _, move = minimax(3, -math.inf, math.inf, player_choice != -1)
    if move:
        row, col = move
        place_stone(row, col, -player_choice)
        return check_winner(-player_choice)
    return False

async def main():
    global current_player, player_choice, game_started
    setup()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if not game_started:
                    # 흑돌 버튼
                    black_button = pygame.Rect(BOARD_WIDTH//4, BOARD_HEIGHT//2 - 50, 150, 50)
                    # 백돌 버튼
                    white_button = pygame.Rect(BOARD_WIDTH//2 + 50, BOARD_HEIGHT//2 - 50, 150, 50)
                    if black_button.collidepoint(x, y):
                        player_choice = 1
                        current_player = 1
                        game_started = True
                        draw_board()
                    elif white_button.collidepoint(x, y):
                        player_choice = -1
                        current_player = 1
                        game_started = True
                        draw_board()
                        if ai_move():
                            print("AI wins!")
                            running = False
                    continue
                if current_player == player_choice:
                    col = round((x - CELL_SIZE / 2) / CELL_SIZE)
                    row = round((y - CELL_SIZE / 2) / CELL_SIZE)
                    if place_stone(row, col, current_player):
                        if check_winner(current_player):
                            print(f"{'Player' if current_player == player_choice else 'AI'} wins!")
                            running = False
                        else:
                            current_player = -current_player
                            if current_player != player_choice and ai_move():
                                print("AI wins!")
                                running = False
                            current_player = -current_player
        await asyncio.sleep(1.0 / FPS)
    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())