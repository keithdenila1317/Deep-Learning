
#####################################################################
# This script presents how to use Doom's native demo mechanism to
# replay episodes with perfect accuracy.
#####################################################################

import os
import vizdoom as vzd
import numpy as np 
import shutil
from random import choice
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Skill Level Folders {Novice/Expert}
skill_level = "Novice"


target_recording = os.path.join(BASE_DIR, "Recordings", skill_level)

game = vzd.DoomGame()

game.load_config("custom_deathmatch.cfg")
wad_path = os.path.join(vzd.scenarios_path, "deathmatch.wad")
game.set_doom_scenario_path(wad_path)
game.set_episode_timeout(10500)

game.set_screen_resolution(vzd.ScreenResolution.RES_512X384)
game.set_render_hud(False)

game.set_mode(vzd.Mode.ASYNC_SPECTATOR)

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

# --- FORCE MODERN BINDING ---

game.send_game_command("bind w +forward")
game.send_game_command("bind s +back")
game.send_game_command("bind a +moveleft")
game.send_game_command("bind d +moveright")

if not os.path.exists(target_recording) :
    os.makedirs(target_recording)

max_count = -1

for filename in os.listdir(target_recording) :
    match = re.search(r'episode(\d+)_rec\.lmp', filename)
    if match :
        cont_num = int(match.group(1))
        if cont_num > max_count :
            max_count = cont_num
        
start_index = max_count + 1

# Number of recordings per session
episodes = 5

print("\nRECORDING EPISODES")
print("************************\n")

for i in range(start_index, start_index + episodes) :

    game.new_episode(f"episode{i}_rec.lmp")

    while not game.is_episode_finished() :

        s = game.get_state()

        game.advance_action() 

        print(f"State #{s.number} recorded!")

    print(f"Episode {i} finished. Saved to file episode{i}_rec.lmp")
    print("Total reward:", game.get_total_reward())
    print("************************\n")

game.new_episode()  
game.close()

# Downsampled rendering for replay
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
game.set_render_hud(True)

game.set_mode(vzd.Mode.SPECTATOR)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")

for i in range(start_index, start_index + episodes) :

    game.replay_episode(f"episode{i}_rec.lmp")

    while not game.is_episode_finished() :

        s = game.get_state()
        assert s is not None and s.game_variables is not None

        game.advance_action()

        a = game.get_last_action()
        r = game.get_last_reward()

        print(f"State #{s.number}")
        print("Action:", a)
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print("Episode", i, "finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")

game.close()

for i in range(start_index, start_index + episodes):
   s_file = f"episode{i}_rec.lmp"

   if os.path.exists(s_file):
        shutil.move(s_file, target_recording)
        print(f"Moved: {s_file}")
   else:
        print(f"Warning: Could not find {s_file}")

   print("All files successfully organized!")
