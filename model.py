import torch as T
import torch.nn as nn #Handles layers (not convolutional)
import torch.nn.functional as F #RELU function
import torch.optim as optim #optimiser
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__() #calls constructor for base class 
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) #star operator unpacks n dimension vector as first args
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) #send entire net to device 
        
    def forward(self, state): #handles forward propagation
        tmp_action = F.relu(self.fc1(state))
        tmp_action = F.relu(self.fc2(tmp_action))
        actions = self.fc3(tmp_action) #returns actions (raw estimate as it's not activated)
        return actions
    
class Agent():
    def __init__(self, gamma, epsilon, learning_rate, input_dims, batch_size, n_actions, max_mem=100000, eps_min=0.01, eps_decr=5e-4):
        self.gamma = gamma #discount factor
        self.epsilon = epsilon #exploration rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size #amount of memories per batch
        self.eps_min = eps_min #how low does exploration rate go
        self.eps_decr = eps_decr #how much does exploration rate decrement per time step
        self.action_space = [i for i in range(n_actions)] #int representation of available actions (for use in greedy selection)
        self.mem_size = max_mem
        self.memory_counter = 0 #track first available memory
        self.q_eval = DeepQNetwork(self.learning_rate, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256) #DeepQNetwork
        self.state_mem = np.zeros((self.mem_size, *input_dims),dtype=np.float32) #memory storage
        self.new_state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32) #used in temporal difference, compare new state to previous state
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.r_mem = np.zeros(self.mem_size, dtype=np.float32) #reward memory
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool) #value of terminal state always 0, track that here ('done' flags will be used, so hence np.bool is used)
         
    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.mem_size #gets first available position in memory, when mem full, overwrite old memories with new ones 
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.r_mem[index] = reward
        self.action_mem[index] = action
        self.terminal_mem[index] = done
        
        self.memory_counter+=1

    def choose_action(self, obsv):
        if np.random.random() > self.epsilon: #greedy choice
            state = T.tensor([obsv]).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else: #explorative move
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self): 
        if self.memory_counter < self.batch_size:
            return
        #start learning as soon as batch size is filled
        self.q_eval.optimizer.zero_grad()
        max_mem = min(self.memory_counter, self.mem_size) #select up to last filled memory
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.float32)
        state_batch = T.tensor(self.state_mem[batch]).to(self.q_eval.device)
        new_state_batch = T.tensor(self.new_state_mem[batch]).to(self.q_eval.device)
        reward_batch = T.tensor(self.r_mem[batch]).to(self.q_eval.device)
        terminal_batch = T.tensor(self.terminal_mem[batch]).to(self.q_eval.device)
        action_batch = self.action_mem[batch]

        q_eval = self.q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.q_eval.forward(new_state_batch) #can implement target network here if required

        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] #returns value and index, get value

        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.epsilon = self.epsilon - self.eps_decr if self.epsilon > self.eps_min else self.eps_min