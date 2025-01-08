import torch


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int, state_dim: tuple[int, ...]):
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.index: int = 0

        self.state_memory = torch.zeros((buffer_size, *state_dim), dtype=torch.float32)
        self.next_state_memory = torch.zeros((buffer_size, *state_dim), dtype=torch.float32)
        self.action_memory = torch.zeros(buffer_size, dtype=torch.int64)
        self.reward_memory = torch.zeros(buffer_size, dtype=torch.float32)
        self.done_memory = torch.zeros(buffer_size, dtype=torch.float32)

        self.length = 0

    def store_transition(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.state_memory[self.index] = state
        self.next_state_memory[self.index] = next_state
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.done_memory[self.index] = done

        self.index = (self.index + 1) % self.buffer_size
        self.length = min(self.length + 1, self.buffer_size)

    def sample_buffer(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        buffer_size = min(self.length, self.buffer_size)
        batch = torch.randint(0, buffer_size, (self.batch_size,))

        return self.state_memory[batch], self.action_memory[batch], self.reward_memory[batch], self.next_state_memory[
            batch], self.done_memory[batch]

    def reset(self):
        self.index = 0
        self.state_memory = torch.zeros_like(self.state_memory)
        self.next_state_memory = torch.zeros_like(self.next_state_memory)
        self.action_memory = torch.zeros_like(self.action_memory)
        self.reward_memory = torch.zeros_like(self.reward_memory)
        self.done_memory = torch.zeros_like(self.done_memory)

    def __len__(self):
        return self.length
