# generates Pseudo demo for training 
# unused
import torch
import numpy as np
import random
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tasks2d import LousyPacmanPseudoGame as PseudoGame 
from configs import PSEUDO_SCREEN_HEIGHT, PSEUDO_SCREEN_WIDTH
from tasks2d import LousyPacmanPseudoGameEasy as PseudoGameEasy



class PseudoDemoGeneratorEasy:

    def __init__(self, device, num_demos=5, min_num_waypoints=2, max_num_waypoints=6, 
                 num_threads=2, demo_length = 10):
        self.num_demos = num_demos
        self.min_num_waypoints = min_num_waypoints
        self.max_num_waypoints = max_num_waypoints
        self.device = device
        self.agent_key_points = PseudoGame.agent_keypoints
        self.translation_scale = 500
        self.demo_length = demo_length

        self.player_speed = 5 
        self.player_rot_speed = 5
        self.num_threads = num_threads
        self.biased_odds = 0.1
        self.augmented = True
        
        # Thread-local storage for agent keypoints
        self._thread_local = threading.local()

    def get_batch_samples(self, batch_size: int) -> Tuple[torch.Tensor, List, torch.Tensor]:
        """
        Generate a batch of samples in parallel
        Returns:
            curr_obs_batch: List of batch_size current observations
            context_batch: List of batch_size contexts (each context is a list of demos)
            clean_actions_batch: Tensor of shape [batch_size, pred_horizon, 4]
        """
        # Use ThreadPoolExecutor to generate samples in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all sample generation tasks

            futures = [executor.submit(self._generate_single_sample) for _ in range(batch_size)]
            
            # Collect results as they complete
            curr_obs_batch = []
            context_batch = []
            clean_actions_list = []

            
            for future in as_completed(futures):
                curr_obs, context, clean_actions = future.result()
                curr_obs_batch.append(curr_obs)
                context_batch.append(context)
                clean_actions_list.append(clean_actions)

        
        # Stack clean actions into a single tensor [batch_size, pred_horizon, 4]
        # convert
        clean_actions_batch = clean_actions_list
        
        return curr_obs_batch, context_batch, clean_actions_batch

    def _generate_single_sample(self) -> Tuple[dict, List, torch.Tensor]:
        """Generate a single training sample (thread-safe)"""
        biased = np.random.rand() < self.biased_odds
        augmented = self.augmented # for now 
        pseudo_game = self._make_game(biased, augmented)
        context = self._get_context(pseudo_game)   
        curr_obs, clean_actions = self._get_ground_truth(pseudo_game)
        return curr_obs, context, clean_actions

    def get_agent_keypoints(self):
    
        agent_keypoints = torch.zeros((len(self.agent_key_points), 2), device=self.device)
        agent_keypoints[0] = torch.tensor(self.agent_key_points['front'], device=self.device)
        agent_keypoints[1] = torch.tensor(self.agent_key_points['back-left'], device=self.device)
        agent_keypoints[2] = torch.tensor(self.agent_key_points['back-right'], device=self.device)
        agent_keypoints[3] = torch.tensor(self.agent_key_points['center'], device=self.device)
        return agent_keypoints
    
    def _make_game(self, biased,augmented):
        player_starting_pos =(random.randint(0,PSEUDO_SCREEN_WIDTH), random.randint(0,PSEUDO_SCREEN_HEIGHT))
        return PseudoGameEasy(
                    player_starting_pos=player_starting_pos,
                    max_num_sampled_waypoints=self.max_num_waypoints, 
                    min_num_sampled_waypoints=self.min_num_waypoints, 
                    biased=biased, 
                    augmented=augmented
                )

    def _run_game(self, pseudo_demo):
        max_retries = 1000
        # player_starting_pos =(random.randint(0,PSEUDO_SCREEN_WIDTH), random.randint(0,PSEUDO_SCREEN_HEIGHT))
        for attempt in range(max_retries):
            try: 
                # first reset 
                pseudo_demo.reset_game(shuffle=True) # config stays, but game resets (player, obj change positions)
                pseudo_demo.run()
                if len(pseudo_demo.actions) < 2:
                    continue
                return pseudo_demo
            except Exception as e:
                if attempt == max_retries-1:
                    raise 
                continue

    def _get_context(self, pseudo_game):
        context = []
        for _ in range(self.num_demos - 1):
            pseudo_demo = self._run_game(pseudo_game)
            observations = pseudo_demo.observations
            sampled_obs = self._downsample_obs(observations)
            context.append(sampled_obs)
        return context
            
    def _get_ground_truth(self, pseudo_game):
        pseudo_game.set_augmented(np.random.rand() > 0.5) 
        pseudo_demo = self._run_game(pseudo_game)
        pd_actions = pseudo_demo.get_actions(mode='vector')

        se2_actions = np.array([action[0].flatten() for action in pd_actions]).reshape(-1, 9) # n x 9
        state_actions = np.array([action[1] for action in pd_actions]) # n x 1
        state_actions = state_actions.reshape(-1,1)
        actions = np.concatenate([se2_actions, state_actions], axis=1)
        actions = torch.tensor(
            actions, 
            dtype=torch.float, 
            device=self.device
        )          
        temp = actions.shape
        actions = self._accumulate_actions(actions)
        sample_rate = min(actions.shape[0] // self.demo_length,1)
        assert temp == actions.shape
        actions = self._downsample_actions(actions)
        true_obs = self._downsample_obs(pseudo_demo.observations)


        return true_obs[0], actions
    
    def _accumulate_actions(self, actions):
        n = actions.shape[0]
        
        # Extract and reshape SE(2) matrices
        se2_matrices = actions[:, :9].view(n, 3, 3)
        state_actions = actions[:, 9:]
        
        # Compute cumulative matrix products
        cumulative_matrices = torch.zeros_like(se2_matrices)
        cumulative_matrices[0] = se2_matrices[0]
        
        for i in range(1, n):
            cumulative_matrices[i] = torch.matmul(cumulative_matrices[i-1], se2_matrices[i])
        
        # Flatten back and concatenate with state actions
        cumulative_se2_flat = cumulative_matrices.view(n, 9)
        cumulative_actions = torch.cat([cumulative_se2_flat, state_actions], dim=1)
        
        return cumulative_actions

    def _downsample_actions(self,actions):
        if actions.shape[0] < self.demo_length:
            return actions
        result = torch.zeros((10, actions.shape[1]), device = actions.device)
        result[0] = actions[0]
        total_middle_positions = actions.shape[0] - 2  # Exclude first and last
        
        for i in range(1, self.demo_length - 1):
            # Calculate the position in the middle section (0 to total_middle_positions-1)
            middle_pos = (i - 1) * total_middle_positions / (self.demo_length - 3) if self.demo_length > 3 else total_middle_positions // 2
            # Convert to actual index (add 1 to skip first item)
            actual_index = int(round(middle_pos)) + 1
            result[i] = actions[actual_index]
        
        result[-1] = actions[-1]  # Always include last item
        
        return result
    
    def _downsample_obs(self, observations):
        if len(observations) < self.demo_length:
            return observations

        result = [observations[0]]
        total_middle_positions = len(observations) - 2  # Exclude first and last
        
        for i in range(1, self.demo_length - 1):
            # Calculate the position in the middle section (0 to total_middle_positions-1)
            middle_pos = (i - 1) * total_middle_positions / (self.demo_length - 3) if self.demo_length > 3 else total_middle_positions // 2
            # Convert to actual index (add 1 to skip first item)
            actual_index = int(round(middle_pos)) + 1
            result.append(observations[actual_index])
        
        result.append(observations[-1])  # Always include last item
        
        return result
