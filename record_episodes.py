#!/usr/bin/env python3

#####################################################################
# This script presents how to use Doom's native demo mechanism to
# replay episodes with perfect accuracy.
#####################################################################

import os
from random import choice

import vizdoom as vzd
import numpy as np 
import shutil

# 1. This magically finds the folder where record_episodes.py is currently sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the path dynamically for the specific player/skill
skill_level = "Novice"
target_recording = os.path.join(BASE_DIR, "Recordings", skill_level)

game = vzd.DoomGame()

# Use other config file if you wish.
game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
game.set_episode_timeout(100)

# Record episodes while playing in 320x240 resolution without HUD
game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
game.set_render_hud(False)

# Episodes can be recorder in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
game.set_mode(vzd.Mode.ASYNC_SPECTATOR)

game.init()

actions = [[True, False, False], [False, True, False], [False, False, True]]

# Run and record this many episodes
episodes = 3

# Recording
print("\nRECORDING EPISODES")
print("************************\n")

for i in range(episodes):

    # new_episode can record the episode using Doom's demo recording functionality to given file.
    # Recorded episodes can be reconstructed with perfect accuracy using different rendering settings.
    # This can not be used to record episodes in multiplayer mode.
    game.new_episode(f"episode{i}_rec.lmp")

    while not game.is_episode_finished():
        s = game.get_state()
        
        # Just advance the frame. YOU press the keys!
        game.advance_action() 

        print(f"State #{s.number} recorded!")

    print(f"Episode {i} finished. Saved to file episode{i}_rec.lmp")
    print("Total reward:", game.get_total_reward())
    print("************************\n")

game.new_episode()  # This is currently required to stop and save the previous recording.
game.close()

# New render settings for replay
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
game.set_render_hud(True)

# Replay can be played in any mode.
game.set_mode(vzd.Mode.SPECTATOR)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")

for i in range(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game.replay_episode(f"episode{i}_rec.lmp")

    while not game.is_episode_finished():
        # Get a state
        s = game.get_state()
        assert s is not None and s.game_variables is not None

        # Use advance_action instead of make_action to proceed
        game.advance_action()

        # Retrieve the last actions and the reward
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

for i in range(episodes):
   s_file = f"episode{i}_rec.lmp"

   if os.path.exists(s_file):
        shutil.move(s_file, target_recording)
        print(f"Moved: {s_file}")
   else:
        print(f"Warning: Could not find {s_file}")

   print("All files successfully organized!")
