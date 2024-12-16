from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

def decode(obs: np.ndarray):
    print(obs[:144].reshape(12, 12))
    print(obs[144:144*2].reshape(12, 12))
    print(obs[144*2:144*3].reshape(12, 12))
    print(obs[144*3:144*4].reshape(12, 12))
    print(obs[144*4:])
    
    

class MCTSNode:
    def __init__(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        reward: float,
        value: float,
        done: bool,
        prob: float,
        parent=None
    ) -> None:
        self.state = state
        self.mask = mask
        self.visits = 0
        
        self.reward = reward
        self.value = value # value of this node after coming from its parent
        self.prob = prob # probability of visiting this node from the parent
        self.done = done
        
        
        self.Q = 0
        self.parent = parent
        self.children = {}
        self.action_probs = None

    def __repr__(self):
        return f"MCTSNode (state={self.state.shape})"


class MCTS(nn.Module):
    def __init__(self, model: nn.Module, env: gym.Env, args) -> None:
        super().__init__()
        self.model = model
        self.env = env
        self.num_simulations = args.num_simulations
        self.exploration_constant = args.exploration_constant
        self.args = args

    def search(self, root_state: torch.Tensor, root_mask: torch.Tensor) -> np.ndarray:
        
        
        roots = []
        root_state = root_state.cpu().numpy()
        root_mask = root_mask.cpu().numpy()
        for env_id in range(root_state.shape[0]):
            root = MCTSNode(root_state[env_id], root_mask[env_id], reward=0, value=0, done=False, prob=1, parent=None)
            roots.append(root)
        
        for _ in range(self.num_simulations):
            
            nodes = [self.select(root) for root in roots]
            
            masks = torch.Tensor([node.mask for node in nodes]).to(self.model_device).bool()            
            states = torch.Tensor([node.state for node in roots]).to(self.model_device)
            
            with torch.no_grad():
                _, _, _, value, action_probs = self.model.get_action_and_value(states, masks)
            
            for i, node in enumerate(nodes):
                node.action_probs = action_probs[i].cpu().numpy()
                node.value = value[i].cpu().numpy()
                
                node = self.expand(node)
                self.backpropagate(node)
        
        best_action = np.zeros((root_state.shape[0]))
        
        for env_id in range(root_state.shape[0]):
            
            best_action[env_id] = self.best_child(roots[env_id])
        
        return best_action  # Return the action leading to the best child


    def select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            action, node = self.best_uct(node)
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:

        # chosen_action = node.action_probs.argmax(axis=-1)
        for action, prob in enumerate(node.action_probs):
            if node.mask[action] == 0:
                continue
            
            chosen_action = action
            # decode(node.state)
            # print(chosen_action)
            
            next_state, reward, terminated, truncated, info = self.env.step_logic(node.state.copy(), chosen_action, node.mask.copy())
            # decode(next_state)
            # assert 0
            child = MCTSNode(next_state, info["avail_actions"], reward, value=None, done=terminated or truncated, prob=prob, parent=node)
            node.children[action] = child
        
        return node


    def backpropagate(self, node: MCTSNode) -> None:
        current_value = node.value.sum()
        while node:
            node.Q = (node.Q * node.visits + current_value) / (node.visits + 1)
            current_value = current_value * self.args.gamma + node.reward
            node.visits += 1
            node = node.parent

    def best_child(self, node: MCTSNode) -> int:
        # best_action, best_child = max(node.children.items(), key=lambda item: item[1].N)
        prob_dist = np.array([child.visits for child in node.children.values()])
        
        # use softmax to get the probability distribution
        prob_dist = np.exp(prob_dist) / np.sum(np.exp(prob_dist))
    
        # sample from the distribution to get the action
        dist = torch.distributions.Categorical(torch.from_numpy(prob_dist))
        
        child_ids = np.array(list(node.children.keys()))
        
        action = dist.sample().item() 
        # action = dist.probs.argmax().item()
        
        return child_ids[action]


    def best_uct(self, node: best_child) -> Tuple[int, MCTSNode]:
        children = [(action, child) for action, child in node.children.items()]
        uct_values = [(action, self.uct_value(child)) for action, child in children]
        
        action, uct_value = max(uct_values, key=lambda item: item[1])
        child = node.children[action]
        return action, child

    def uct_value(self, node: best_child) -> float:
        exploit = node.Q/100.0
        explore = node.prob * np.sqrt(node.parent.visits) / (1 + node.visits) * np.sqrt(2)
        
        return exploit + self.exploration_constant * explore

    def get_action_and_value(self, state: torch.Tensor, mask: torch.Tensor, action: torch.Tensor | None = None) -> Tuple[torch.Tensor, ...]:
        self.model_device = self.model.parameters().__next__().device
        
        if self.args.num_simulations > 0: # Use MCTS if num_simulations > 0
        
            if action is None:
                action = self.search(state.clone(), mask.clone())
                action = torch.from_numpy(action).to(self.model_device).long()
                
            # check actions are not masked
            assert (mask.gather(-1, action.unsqueeze(-1)) == 1).all()

        action, log_prob, entropy, value, probs = self.model.get_action_and_value(state, mask, action)
            
        return action, log_prob, entropy, value, probs
        
    def get_eval_action(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        self.model_device = self.model.parameters().__next__().device
        if self.args.num_simulations > 0: # Use MCTS if num_simulations > 0
            action = self.search(x.clone(), action_mask.clone())
            action = torch.from_numpy(action).to(self.model_device).long()
                
            # check actions are not masked
            assert (action_mask.gather(-1, action.unsqueeze(-1)) == 1).all()
        else:
            action = self.model.get_eval_action(x, action_mask)
        return action
            
            