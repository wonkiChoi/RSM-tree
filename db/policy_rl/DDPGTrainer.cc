/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <math.h>
#include "DDPGTrainer.h"

/******************* Actor *******************/
Actor::Actor(int64_t state_size, int64_t action_size) : torch::nn::Module() {
  conv1 = register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(state_size, 32, 2).stride(1)));
  conv2 = register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 2).stride(1)));
  linear1 = register_module("linear1", torch::nn::Linear(128, 64));
  output = register_module("output", torch::nn::Linear(64, action_size));
}

torch::Tensor Actor::forward(torch::Tensor input) {
  input = input.transpose(1, 2);
  input = torch::relu(conv1(input));
  input = torch::relu(conv2(input));

  // Flatten the output
  input = input.view({input.size(0), -1});
  input = torch::relu(linear1(input));
  input = output(input);
  input = torch::sigmoid(input);

  return input;
}

/******************* Critic *****************/
Critic::Critic(int64_t state_size, int64_t action_size) : torch::nn::Module() {
  conv1 = register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(state_size, 32, 2).stride(1)));
  conv2 = register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 2).stride(1)));
  linear1 = register_module("linear1", torch::nn::Linear(64*2, 64));
  
  fc1 = register_module("fc1", torch::nn::Linear(64 + 1, 32));
  fc2 = register_module("fc2", torch::nn::Linear(32, 1));
}

torch::Tensor Critic::forward(torch::Tensor input, torch::Tensor action) {
  if (action.dim() == 1) action = torch::unsqueeze(action, 0);
  input = input.transpose(1, 2);
  input = torch::relu(conv1(input));
  input = torch::relu(conv2(input));

  input = input.view({input.size(0), -1});
  input = torch::relu(linear1(input));

  auto x = torch::cat({input, action}, /*dim=*/1);
  x = torch::relu(fc1->forward(x));
  return fc2->forward(x);
}

DDPGTrainer::DDPGTrainer(int64_t input_channels, int64_t num_actions, int64_t capacity)
    : Trainer(capacity), actor_optimizer(actor_local->parameters(), lr_actor), critic_optimizer(critic_local->parameters(), lr_critic){
    stateSize = input_channels;
    actionSize = num_actions;
    
    actor_local = std::make_shared<Actor>(stateSize, actionSize);
    actor_target = std::make_shared<Actor>(stateSize, actionSize);
  
    critic_local = std::make_shared<Critic>(stateSize, actionSize);
    critic_target = std::make_shared<Critic>(stateSize, actionSize);

    actor_local->to(torch::Device(torch::kCPU));
    actor_target->to(torch::Device(torch::kCPU));

    critic_local->to(torch::Device(torch::kCPU));
    critic_target->to(torch::Device(torch::kCPU));

    critic_optimizer.options.weight_decay_ = weight_decay;

    hard_copy(actor_target, actor_local);
    hard_copy(critic_target, critic_local);
    noise = new OUNoise(static_cast<size_t>(actionSize));   
}  

std::vector<double> DDPGTrainer::act(std::vector<double> state) {
  torch::Tensor torchState = torch::tensor(state, torch::dtype(torch::kDouble)).to(torch::kCPU);
  actor_local->eval();

  torch::NoGradGuard guard;
  torch::Tensor action = actor_local->forward(torchState).to(torch::kCPU);
  actor_local->train();
  std::vector<double> v(action.data_ptr<double>(), action.data_ptr<double>() + action.numel());
  noise->sample(v);
  
  for (size_t i =0; i < v.size(); i++) {
    v[i] = std::fmin(std::fmax(v[i],-1.f), 1.f);
  }
  return v;
}

void DDPGTrainer::learn() {
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
    buffer.sample_queue(batch_size);
  std::vector<torch::Tensor> states;
  std::vector<torch::Tensor> new_states;
  std::vector<torch::Tensor> actions;
  std::vector<torch::Tensor> rewards;

  for (auto i : batch) {
    states.push_back(std::get<0>(i));
    new_states.push_back(std::get<1>(i));
    actions.push_back(std::get<2>(i));
    rewards.push_back(std::get<3>(i));
  }

  torch::Tensor states_tensor = torch::cat(states, 0);
  torch::Tensor new_states_tensor = torch::cat(new_states, 0);
  torch::Tensor actions_tensor = torch::cat(actions, 0);
  torch::Tensor rewards_tensor = torch::cat(rewards, 0);
                    
  auto actions_next = actor_target->forward(new_states_tensor);
  auto Q_targets_next = critic_target->forward(new_states_tensor, actions_next);
  auto Q_targets = rewards_tensor + (gamma * Q_targets_next);
  auto Q_expected = critic_local->forward(states_tensor, actions_tensor); 

  torch::Tensor critic_loss = torch::mse_loss(Q_expected, Q_targets.detach());
  critic_optimizer.zero_grad();
  critic_loss.backward();
  critic_optimizer.step();

  auto actions_pred = actor_local->forward(states_tensor);
  auto actor_loss = -critic_local->forward(states_tensor, actions_pred).mean();

  actor_optimizer.zero_grad();
  actor_loss.backward();
  actor_optimizer.step();

  soft_update(critic_local, critic_target);
  soft_update(actor_local, actor_target);    
}

void DDPGTrainer::soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i].copy_(tau * local->parameters()[i] + (1.0 - tau) * target->parameters()[i]);
  }   
}

void DDPGTrainer::hard_copy(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i] = local->parameters()[i];
  }
}

