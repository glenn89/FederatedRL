[CartPole-v0]
- https://github.com/openai/gym/wiki/CartPole-v0

- Action
  - 0: Left
  - 1: Right

- Reward
  - Reward is 1 for every step taken, including the termination step

- Starting State
  - All observations are assigned a uniform random value between ±0.05

- Episode Termination
  - Pole Angle is more than ±12°
  - Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
  - Episode length is greater than 200

- Solved Requirements
  - Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
  
[RoboschoolAnt-v1]
a four-legged agent called the Ant.
The body is supported by 4 legs, and each leg consists of 3 parts which are controlled by 2 motor joints.
Goal: move as fast as possible, minimizing the energy spent.

foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
action_dim=8,
obs_dim=28,
power=2.5

def alive_bonus(self, z, pitch):
    return +1 if z > 0.26 else -1  # 0.25 is central sphere radian, die if it scrapes the ground
