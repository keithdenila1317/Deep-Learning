
#####################################################################
# Script to convert .lmp files previously recorded into numpy array
# taking two variables for action (key) and state (image). 
#####################################################################

import vizdoom as vzd
import numpy as np
import os
import glob

skill_level = "Recordings/Novice"
output = "novice_dataset.npz"

frames = 4

def extract() :
    
    lmp_files = glob.glob(os.path.join(skill_level, "*.lmp"))

    if not lmp_files :
        print("No .lmp files")
    
    print(f"{len(lmp_files)} episodes to convert")

    game = vzd.DoomGame()
    game.load_config("custom_deathmatch.cfg")
    wad_path = os.path.join(vzd.scenarios_path, "deathmatch.wad")
    game.set_doom_scenario_path(wad_path)

    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_window_visible(True)

    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()

    total_frames = []
    total_actions = []

    for episode in lmp_files :

        game.replay_episode(episode)
        frame_count = 0

        while not game.is_episode_finished() :

            state = game.get_state()
            game.advance_action()
            action = game.get_last_action()

            if state is not None and frame_count % frames == 0 :

                current_frame = state.screen_buffer
                total_frames.append(current_frame)
                total_actions.append(action)

            frame_count += 1
    
    game.close()

    new_frames = np.array(total_frames, dtype = np.uint8)
    new_actions = np.array(total_actions, dtype = np.float32)

    if os.path.exists(output) :
        
        old_data = np.load(output)
        old_frames = old_data['frames']
        old_actions = old_data['actions']

        final_frames = np.concatenate((old_frames, new_frames), axis = 0)
        final_actions = np.concatenate((old_actions, new_actions), axis = 0)

    else:

        final_frames = new_frames
        final_actions = new_actions

    np.savez_compressed(output, frames = final_frames, actions = final_actions)

if __name__ == "__main__" :

    extract()
