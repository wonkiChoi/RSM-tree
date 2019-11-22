//
// Created by Navneet Madhu Kumar on 2019-07-10.
//
#include "Trainer.h"
#include "dqn.h"
#include "ExperienceReplay.h"
#include <math.h>
#include <chrono>


Trainer::Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity, int64_t frame_id_, int64_t previous_action_):
    buffer(capacity),
    network(input_channels, num_actions),
    target_network(input_channels, num_actions),
    dqn_optimizer(network.parameters(), torch::optim::AdamOptions(0.0001).beta1(0.5)),
    frame_id(frame_id_),
    previous_action(previous_action_){}

    torch::Tensor Trainer::compute_td_loss(int64_t batch_size_, float gamma_){
        std::cout << "compute_td_loss" << std::endl;
      std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
        buffer.sample_queue(batch_size_);

        std::vector<torch::Tensor> states;
        std::vector<torch::Tensor> new_states;
        std::vector<torch::Tensor> actions;
        std::vector<torch::Tensor> rewards;

        for (auto i : batch){
            states.push_back(std::get<0>(i));
            new_states.push_back(std::get<1>(i));
            actions.push_back(std::get<2>(i));
            rewards.push_back(std::get<3>(i));
        }


        torch::Tensor states_tensor;
        torch::Tensor new_states_tensor;
        torch::Tensor actions_tensor;
        torch::Tensor rewards_tensor;

        states_tensor = torch::cat(states, 0);
        new_states_tensor = torch::cat(new_states, 0);
        actions_tensor = torch::cat(actions, 0);
        rewards_tensor = torch::cat(rewards, 0);


        std::cout << "forward " <<std::endl;
        torch::Tensor q_values = network.forward(states_tensor);
        std::cout << "target_ forward " <<std::endl;
        torch::Tensor next_target_q_values = target_network.forward(new_states_tensor);
        std::cout << "next forward " <<std::endl;
        torch::Tensor next_q_values = network.forward(new_states_tensor);

        std::cout << "action " <<std::endl;
        actions_tensor = actions_tensor.to(torch::kInt64);

        std::cout << "q " <<std::endl;
        torch::Tensor q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1);
        std::cout << "q1 " <<std::endl;
        torch::Tensor maximum = std::get<1>(next_q_values.max(1));
        std::cout << "q2 " <<std::endl;
        torch::Tensor next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1);
        std::cout << "q3 " <<std::endl;
        torch::Tensor expected_q_value = rewards_tensor + gamma*next_q_value;
        std::cout << "q4 " <<std::endl;
        torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);
        std::cout << "loss " <<std::endl;

        dqn_optimizer.zero_grad();
        loss.backward();
        dqn_optimizer.step();

        std::cout << "step over " <<std::endl;
        return loss;

    }

    double Trainer::epsilon_by_frame(){
        return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * frame_id / epsilon_decay);
    }

    torch::Tensor Trainer::get_tensor_observation(std::vector<int64_t> state) {
        torch::Tensor state_tensor = torch::from_blob(state.data(), {1, 3, num_level, max_file_num});
//        std::cout << state_tensor << std::endl;
        return state_tensor;
    }

    void Trainer::loadstatedict(torch::nn::Module& model,
                       torch::nn::Module& target_model) {
        torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
        auto new_params = target_model.named_parameters(); // implement this
        auto params = model.named_parameters(true /*recurse*/);
        auto buffers = model.named_buffers(true /*recurse*/);
        for (auto& val : new_params) {
            auto name = val.key();
            auto* t = params.find(name);
            if (t != nullptr) {
                t->copy_(val.value());
            } else {
                t = buffers.find(name);
                if (t != nullptr) {
                    t->copy_(val.value());
                }
            }
        }
    }

    void Trainer::train(int64_t random_seed, std::string rom_path, int64_t num_epochs){
//        load_enviroment(random_seed, rom_path);
//        //ActionVect legal_actions = ale.getLegalActionSet();
//        // ale.reset_game();
//        std::vector<unsigned char> state;
//        //ale.getScreenRGB(state);
//        float episode_reward = 0.0;
//        std::vector<float> all_rewards;
//        std::vector<torch::Tensor> losses;
//        auto start = std::chrono::high_resolution_clock::now();
//        for(int i=1; i<=num_epochs; i++){
//            double epsilon = epsilon_by_frame(i);
//            auto r = ((double) rand() / (RAND_MAX));
//            torch::Tensor state_tensor = get_tensor_observation(state);
//            //Action a;
//            if (r <= epsilon){
//              //  a = legal_actions[rand() % legal_actions.size()];
//            }
//            else{
//                torch::Tensor action_tensor = network.act(state_tensor);
//                int64_t index = action_tensor[0].item<int64_t>();
//               // a = legal_actions[index];
//
//            }
//
//            //float reward = ale.act(a);
//            float reward = 0;
//            episode_reward += reward;
//            std::vector<unsigned char> new_state;
//            //ale.getScreenRGB(new_state);
//            torch::Tensor new_state_tensor = get_tensor_observation(new_state);
//            bool done = true;
//            // ale.game_over();
//
//            torch::Tensor reward_tensor = torch::tensor(reward);
//            torch::Tensor done_tensor = torch::tensor(done);
//            done_tensor = done_tensor.to(torch::kFloat32);
//            //torch::Tensor action_tensor_new = torch::tensor(a);
//
//            //buffer.push(state_tensor, new_state_tensor, action_tensor_new, done_tensor, reward_tensor);
//            state = new_state;
//
//            if (done){
//            //    ale.reset_game();
//                //std::vector<unsigned char> state;
//                state.clear();
//            //    ale.getScreenRGB(state);
//                all_rewards.push_back(episode_reward);
//                episode_reward = 0.0;
//            }
//
//            if (buffer.size_buffer() >= 10000){
//                torch::Tensor loss = compute_td_loss(batch_size, gamma);
//                losses.push_back(loss);
//            }
//
//            if (i%1000==0){
//                std::cout<<episode_reward<<std::endl;
//                loadstatedict(network, target_network);
//            }
//
//        }
//        auto stop = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//        std::cout << "Time taken by function: "
//             << duration.count() << " microseconds" << std::endl;
    }



