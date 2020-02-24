# Federated Reinforcement Learning

We try to allow multiple reinforcement learning agents to learn optimal control policy on their own IoT devices of the same type but with slightly different dynamics. For such multiple IoT devices, there is no guarantee that an agent who interacts only with one IoT device and learns the optimal control policy will also control another IoT device well. Therefore, we may need to apply independent reinforcement learning to each IoT device individually, which requires a costly or time-consuming effort.

To solve this problem, we propose a new federated reinforcement learning architecture where each agent working on its independent IoT device shares their learning experience (i.e., the gradient of loss function) with each other, and transfers a mature policy model parameters into other agents. They accelerate its learning process by using mature parameters. We incorporate the Actor-Critic PPO algorithm into each agent in the proposed collaborative architecture and propose an efficient procedure for the gradient sharing and the model transfer.

We use Quanser's Qube-servo 2 as the real device. We also used CartPole from OpenAI Gym as the simulation environment.
Our related paper is "분산 다중 에이전트 기반 연합 강화 학습을 활용한 학습 성능 개선" in Proceedings of KICS Fall Conference 2019, November, Seoul, Korea, 2019.

### Execute the Proposed Federated Reinforcement Learning in Simulation Environment
- Set-up the hyper-parameter
  - modify main_constants.py in "rl_main" folder
    - ex) the number of workers, Whether or not to use the gradient sharing, Whether or not to use the transfer learning, and PPO's hyper-parameters
- Execution
  - python main.py

## Environment Configuration

### 1. Create Environment
- conda create -n rl python=3.6
- conda activate rl
- pip install --upgrade pip
- pip install -r requirements.txt
- pytorch install
  - https://pytorch.org/ reference
- baselines install
  - https://github.com/openai/baselines reference

### 2. OpenAI Gym Install
- git clone https://github.com/openai/gym.git
- cd gym
- pip install -e '.[all]'
  - ignore mujoco error 

### 3. Package Install & requirements.txt Configuration 

- pip freeze > requirements.txt

### 4. Mosquitto Install
- Mosquitto install
  - brew install mosquitto
  - Linux: https://blog.neonkid.xyz/127

- Execute the mosquitto service 
  - For Mac
    - /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf
  - For Linux
    - mosquitto
  
- Test the subscribe
  - mosquitto_sub -h [address] -p [port] -t [topic]
  - mosquitto_sub -h 127.0.0.1 -p 1883 -t "topic"

- Test the publication
  - mosquitto_pub -h [address] -p [port] -t [topic] -m [content]
  - mosquitto_pub -h 127.0.0.1 -p 1883 -t "topic" -m "test messgae"

### 5. Execution
- Execute chief
  - python main_only_chief.py 
- Execute worker
  - python main_only_one_worker.py 
- Execute main for the test 
  - OpenAI Gym CartPole
  - python main.py


### Reference
- https://dbpia.co.kr/journal/articleDetail?nodeId=NODE09277687
- https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbev
- https://arxiv.org/pdf/1709.06009.pdf
- https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12
- https://en.wikipedia.org/wiki/MM_algorithm
- https://drive.google.com/file/d/0BxXI_RttTZAhMVhsNk5VSXU0U3c/view