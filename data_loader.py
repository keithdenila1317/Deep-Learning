import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torchvision.transforms as transformers
import copy
import torchvision.transforms.functional as TF

class Augmentation(Dataset) :
    
    # change is_train when validating with testing
    def __init__ (self, dataset_npz, is_train = True, seq_len = 15, lambda_o = 1.22) :

        # ram issue change np.load to load batches instead of entire array
        data = np.load(dataset_npz)
        self.frames = data['frames']
        self.actions = data['actions']
        self.seq_len = seq_len
        self.is_train = is_train

        # adds noise / differientiates frames 
        self.use_transform = is_train

        self.offsets = [round(i ** lambda_o) for i in range(seq_len)]
        self.max_offset = max(self.offsets)

    def __len__ (self) :
        
        return len(self.frames) - self.max_offset
    
    def __getitem__ (self, index) :

        curr_index = index + self.max_offset

        seq_indices = [curr_index - o for o in self.offsets]
        seq_indices.reverse()

        # normalization with correct channel dimension (x, x, x)
        frame_seq = self.frames[seq_indices]

        frame = torch.tensor(frame_seq, dtype = torch.float32).unsqueeze(1) / 255.0

        # training goes through transformation, will skip when testing
        if self.use_transform :

            brightness = float(torch.empty(1).uniform_(0.8, 1.2))
            contrast = float(torch.empty(1).uniform_(0.8, 1.2))

            frame = TF.adjust_brightness(frame, brightness)
            frame = TF.adjust_contrast(frame, contrast)

        target_action = self.actions[curr_index]

        # Movement (W, A, S, D, and Mouse 1)
        binary_actions = torch.tensor(target_action[:5], dtype = torch.float32)

        # Vertical and Horizontal mouse movement ([up,down], [left,right])
        normalized_mouse = target_action[5:] / 20.0
        float_actions = torch.tensor(normalized_mouse, dtype = torch.float32)

        return frame, binary_actions, float_actions
    
def get_loaders(dataset_npz, batch_size = 32, seq_len = 15) :

    # train/test switch
    data_set = Augmentation(dataset_npz, is_train = True, seq_len = seq_len)

    # split 80-20 can be adjusted
    training_length = int(0.80 * len(data_set))
    validation_length = len(data_set) - training_length

    # splitting the frames into the two sections randomly
    training_split, validation_split = random_split(data_set, [training_length, validation_length])

    # when testing, disable augmenting
    validation_split.dataset = copy.deepcopy(data_set)
    validation_split.dataset.is_train = False
    validation_split.dataset.transform = False

    training_loader = DataLoader(training_split, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(validation_split, batch_size = batch_size, shuffle = False)

    return training_loader, validation_loader

