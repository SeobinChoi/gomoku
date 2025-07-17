"""
Minimal AlphaZero‑style Gomoku (9×9) for M1 Mac
================================================
‑ Board ‑ 9×9 (state channels 2)
‑ NeuralNet ‑ lightweight CNN outputting (policy logits, value)
‑ MCTS ‑ 100 simulations per move, PUCT
‑ Self‑play generates games into a small replay buffer
‑ Training loop with early stopping on rolling win‑rate

Dependencies: torch >= 1.12 (MPS backend), numpy, tqdm
Run: python az_gomoku.py
"""

import math, random, time, os, json, collections
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import os




# ────────────────────────────────────
# Config
# ────────────────────────────────────
BOARD_SIZE        = 9
WIN_CONDITION     = 5  # five‑in‑a‑row
SIMULATIONS       = 100
BUFFER_SIZE       = 10_000
BATCH_SIZE        = 64
EPOCHS_PER_ITER   = 2
SELFPLAY_GAMES    = 25
LR                = 1e-3
DEVICE            = "mps" if torch.backends.mps.is_available() else "cpu"

# ────────────────────────────────────
# Gomoku Environment
# ────────────────────────────────────
class GomokuEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)  # 0 empty, 1 black, ‑1 white
        self.player = 1
        self.history: List[Tuple[int,int]] = []
        self.winner = 0

    def clone(self):
        env = GomokuEnv()
        env.board = self.board.copy()
        env.player = self.player
        env.history = self.history.copy()
        env.winner = self.winner
        return env

    # legal moves list of (r,c)
    def legal_moves(self):
        if self.winner != 0: return []
        coords = np.argwhere(self.board == 0)
        return [tuple(c) for c in coords]

    # play move
    def play(self, move: Tuple[int,int]):
        r,c = move
        assert self.board[r,c] == 0
        self.board[r,c] = self.player
        self.history.append(move)
        if self._check_win(r,c):
            self.winner = self.player
        elif len(self.history) == BOARD_SIZE*BOARD_SIZE:
            self.winner = 2  # draw
        self.player *= -1

    def _check_win(self, r:int, c:int) -> bool:
        for dr,dc in [(1,0),(0,1),(1,1),(1,-1)]:
            cnt = 1
            for sgn in (1,-1):
                nr, nc = r+dr*sgn, c+dc*sgn
                while 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and self.board[nr,nc]==self.player:
                    cnt += 1
                    nr += dr*sgn; nc += dc*sgn
            if cnt >= WIN_CONDITION:
                return True
        return False

    # NN input tensor [2,H,W]
    def tensor(self):
        p = (self.player+1)//2  # 1 if black to move else 0
        plane_current = (self.board ==  self.player).astype(np.float32)
        plane_oppo    = (self.board == -self.player).astype(np.float32)
        return torch.from_numpy(np.stack([plane_current, plane_oppo])).to(DEVICE)

# ────────────────────────────────────
# Neural Network
# ────────────────────────────────────
class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        C = 64
        self.conv1 = nn.Conv2d(2, C, 3, padding=1)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1)
        self.conv3 = nn.Conv2d(C, C, 3, padding=1)
        self.head_policy = nn.Conv2d(C, 2, 1)
        self.fc_policy   = nn.Linear(2*BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE)
        self.head_value  = nn.Conv2d(C, 1, 1)
        self.fc_value    = nn.Linear(BOARD_SIZE*BOARD_SIZE, 1)

    def forward(self, x):  # x [B,2,H,W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # policy
        p = F.relu(self.head_policy(x))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)
        # value
        v = F.relu(self.head_value(x))
        v = v.view(v.size(0), -1)
        v = torch.tanh(self.fc_value(v))
        return p, v.squeeze(1)

# ────────────────────────────────────
# MCTS with PUCT
# ────────────────────────────────────
class TreeNode:
    def __init__(self, prior):
        self.N = 0
        self.W = 0.
        self.Q = 0.
        self.P = prior
        self.children: Dict[int,'TreeNode'] = {}

class MCTS:
    def __init__(self, net:PolicyValueNet):
        self.net = net
        self.c_puct = 1.0

    def search(self, env:GomokuEnv):
        root = TreeNode(0)
        # expand root
        p_logits, _ = self.net(env.tensor().unsqueeze(0))
        probs = F.softmax(p_logits, dim=1).detach().cpu().numpy()[0]
        legal = env.legal_moves()
        for (r,c) in legal:
            idx = r*BOARD_SIZE+c
            root.children[idx] = TreeNode(probs[idx])
        # sims
        for _ in range(SIMULATIONS):
            env_clone = env.clone()
            node = root
            path = []
            # selection
            while node.children:
                best, best_ucb = None, -float('inf')
                sqrt_N = math.sqrt(node.N+1)
                for idx, child in node.children.items():
                    ucb = child.Q + self.c_puct * child.P * sqrt_N/(1+child.N)
                    if ucb > best_ucb:
                        best_ucb, best = ucb, idx
                move_idx = best
                path.append((node, move_idx))
                r,c = divmod(move_idx, BOARD_SIZE)
                env_clone.play((r,c))
                node = node.children[move_idx]
            # expansion
            winner = env_clone.winner
            if winner == 0:
                p_logits, v = self.net(env_clone.tensor().unsqueeze(0))
                probs = F.softmax(p_logits, dim=1).detach().cpu().numpy()[0]
                legal = env_clone.legal_moves()
                node.children = {m_idx:TreeNode(probs[m_idx]) for m_idx in [r*BOARD_SIZE+c for (r,c) in legal]}
                value = v.item()
            elif winner == 2:  # draw
                value = 0
            else:
                value = 1 if winner == env_clone.player*-1 else -1  # from current node perspective
            # backprop
            for parent, move_idx in reversed(path):
                child = parent.children[move_idx]
                child.N += 1
                child.W += value
                child.Q = child.W / child.N
                value = -value  # alternate players
                parent.N += 1  # keep parent N too
        # policy target proportional to visit counts
        pi = np.zeros(BOARD_SIZE*BOARD_SIZE, np.float32)
        for idx, child in root.children.items():
            pi[idx] = child.N
        if pi.sum() == 0:  # no legal? shouldn't happen
            pi += 1
        pi = pi / pi.sum()
        return pi

# ────────────────────────────────────
# Replay Buffer & Dataset
# ────────────────────────────────────
Transition = collections.namedtuple('Transition', 'state pi z')
class ReplayBuffer:
    def __init__(self, size:int):
        self.buffer: List[Transition] = []
        self.size = size
    def add(self, trans:Transition):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(trans)
    def sample(self, batch:int):
        return random.sample(self.buffer, batch)

class BufferDataset(Dataset):
    def __init__(self, samples:List[Transition]):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s,pi,z = self.samples[idx]
        return s, pi, z

# ────────────────────────────────────
# Training loop
# ────────────────────────────────────
net = PolicyValueNet().to(DEVICE)
optim = torch.optim.Adam(net.parameters(), lr=LR)
rb = ReplayBuffer(BUFFER_SIZE)

mcts = MCTS(net)

for iter in range(1, 101):  # max iterations
    print(f"Start training on device")

    # Self‑play games
    for g in range(SELFPLAY_GAMES):
        env = GomokuEnv()
        game_data = []
        while env.winner == 0:
            pi = mcts.search(env)
            rb.add(Transition(env.tensor().cpu(), pi, None))  # value placeholder
            # temperature=1 early moves, greedy later: simple argmax
            move_idx = np.random.choice(len(pi), p=pi)
            env.play(divmod(move_idx, BOARD_SIZE))
        # set z for each transition
        z_final = 0 if env.winner==2 else (1 if env.winner==1 else -1)
        for idx in range(len(rb.buffer)-len(env.history), len(rb.buffer)):
            s, pi, _ = rb.buffer[idx]
            rb.buffer[idx] = Transition(s, pi, z_final if (idx%2==0) else -z_final)

    # Training
    ds = BufferDataset(rb.buffer)
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, len(ds)), shuffle=True)
    for epoch in range(EPOCHS_PER_ITER):
        for states, pis, zs in dl:
            states = states.to(DEVICE)
            pis   = pis.to(DEVICE)
            zs    = zs.to(DEVICE)
            optim.zero_grad()
            p_logits, v = net(states)
            loss_p = F.cross_entropy(p_logits, torch.argmax(pis, dim=1))
            loss_v = F.mse_loss(v, zs.float())
            loss = loss_p + loss_v
            loss.backward()
            optim.step()

    print(f"Iter {iter} | buffer {len(rb.buffer)} | loss {loss.item():.3f}")

    # Early stop: crude — if net beats random 80% over 20 games
    wins = 0
    for _ in range(20):
        env = GomokuEnv()
        while env.winner == 0:
            pi = mcts.search(env)
            move_idx = np.argmax(pi)
            env.play(divmod(move_idx, BOARD_SIZE))
            if env.winner != 0: break
            # random opponent
            move = random.choice(env.legal_moves())
            env.play(move)
        if env.winner == 1:
            wins += 1
    print(f"Test vs random: {wins}/20")
    if wins >= 16:
        print("Early stop: agent good enough! ☕️")
        break
    
def save_model(policy_net, episode, accuracy, is_best=False):
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"az9x9_{'best' if is_best else 'latest'}_ep{episode}_acc{int(accuracy*100)}_date{date_str}.pt"
    save_dir = "logs/weights"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(policy_net.state_dict(), path)
    print(f"✅ Model saved to: {path}")

# 마지막 줄쯤에 추가 — early stop 이후
accuracy = wins / 20  # 평가 승률
save_model(net, iter, accuracy, is_best=True)

