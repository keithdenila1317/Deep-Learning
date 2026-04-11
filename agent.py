import os

import cv2
import torch
import vizdoom as vzd
import numpy as np
from collections import deque
from model import VizDoomCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VizDoomCNN().to(device)

model.load_state_dict(torch.load("expert_model_best.pth", map_location = device))
model.eval()

game = vzd.DoomGame()
game.load_config("custom_deathmatch.cfg")
wad_path = os.path.join(vzd.scenarios_path, "deathmatch.wad")
game.set_doom_scenario_path(wad_path)
game.set_episode_timeout(10500)

game.set_screen_resolution(vzd.ScreenResolution.RES_1920X1080)
game.set_render_hud(False)

game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_mode(vzd.Mode.ASYNC_PLAYER)

# --- STANDARDIZED TEAM SETTINGS ---

# Sensitivity 
game.add_game_args("+set sensitivity 1.5") 
# Mouse Acceleration
game.add_game_args("+set m_customaccel 0") 
# Mouse Smoothing
game.add_game_args("+set m_filter 0") 
# FOV
game.add_game_args("+set fov 90")
# Freelook
game.add_game_args("+freelook 1")

game.add_game_args("+set m_yaw 0.05")

game.init()

episodes = 3
sequence_length = 15

for episode in range(episodes) : 
    game.new_episode()

    frame_buffer = deque(maxlen = sequence_length)

    while not game.is_episode_finished() :

        state = game.get_state()
        screen = state.screen_buffer

        gray = np.mean(screen, axis = 2)
        resized = cv2.resize(gray, (160, 120))

        frame = resized.astype(np.float32) / 255.0

        frame_buffer.append(frame)

        if len(frame_buffer) < sequence_length :

            while len(frame_buffer) < sequence_length :

                frame_buffer.append(frame)

        stacked = np.stack(frame_buffer)

        frame_tensor = torch.tensor(stacked).unsqueeze(0).unsqueeze(2).to(device)

        with torch.no_grad() :
            discrete, continuous = model(frame_tensor)

        probability = torch.sigmoid(discrete)

        #expert values
        #thresholds = torch.tensor([0.35, 0.2, 0.2, 0.5, 0.1]).to(device)
        #thresholds = torch.tensor([0.30, 0.15, 0.08, 0.85, 0.05]).to(device)
        #thresholds = torch.tensor([0.10, 0.25, 0.25, 0.20, 0.35]).to(device)
        #thresholds = torch.tensor([0.08, 0.30, 0.50, 0.05, 0.05]).to(device)
        #thresholds = torch.tensor([0.10, 0.30, 0.30, 0.15, 0.45]).to(device)
        thresholds = torch.tensor([0.05, 0.05, 0.35, 0.10, 0.9995]).to(device)
        buttons = (probability > thresholds).int().cpu().numpy().flatten()

        mouse = continuous.cpu().numpy().flatten() * 350

        if len(mouse) > 1:
            mouse[1] = 0.0

        #if abs(mouse[0]) < 0.5:
        #   mouse[0] = 0.0

        raw_probs = probability.cpu().numpy().flatten()

        print(f"Probs: {np.round(raw_probs, 3)} | Mouse: {np.round(mouse, 3)}")

        action = list(buttons) + list(mouse)

        game.make_action(action)

    print(f"Episode {episode + 1} score: {game.get_total_reward()}")

game.close()