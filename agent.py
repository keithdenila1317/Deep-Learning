import os

import cv2
import torch
import vizdoom as vzd
import numpy as np
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

game.init()

episodes = 5

for episode in range(episodes) : 
    game.new_episode()

    while not game.is_episode_finished() :
        state = game.get_state()
        screen = state.screen_buffer

        gray = np.mean(screen, axis = 2)
        resized = cv2.resize(gray, (160, 120))

        frame = resized.astype(np.float32) / 255.0

        frame_tensor = torch.tensor(frame).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad() :
            discrete, continuous = model(frame_tensor)

        probability = torch.sigmoid(discrete)
        buttons = (probability > 0.5).int().cpu().numpy().flatten()

        mouse = continuous.cpu().numpy().flatten() * 55.0

        action = list(buttons) + list(mouse)

        game.make_action(action)

        print(f"Episode {episode + 1} score: {game.get_total_reward()}")

game.close()