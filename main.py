import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10

# number of times we run the program
HM_EPISODES = 25000
MOVE_PENALTY = 1
# negative reward for running into the enemy
ENEMY_PENALTY = 300
# positive reward for eating the food
FOOD_REWARD = 25
# epsilon creates some randomness so that q does not become over-fitted
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
# we will display only every 3000 frames
SHOW_EVERY = 3000  # how often to play through env visually.

start_q_table = None  # None or Filename

# parameters for learning function
LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# defining player colors. BGR- high [0] = very blue, high [1] = very green, high [2] = very red
# 255 is max color value
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


class Blob:
    def __init__(self):
        # returns 1x10 array of values from low to high, in this case low is 0 and high is 10
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    # string method prints blob's location
    def __str__(self):
        return f"{self.x}, {self.y}"

    # returns distance from blob to other blob
    def __sub__(self, other):
        return self.x-other.x, self.y-other.y

    # we can move along the diagonals
    def action(self, choice):
        self.x = 9
        if choice == 0:
            self.move(x=0, y=1)
        elif choice == 1:
            self.move(x=0, y=-1)

    def f_d_action(self):
        if self.x == 9:
            self.x = 0
            self.y = np.random.randint(0, 10)
        else:
            self.x += 1

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            # will do a -1, 0, or 1 ---> [-1, 1]
            self.x += 0
        else:
            self.x += x

        if not y:
            # will do a -1, 0, or 1 ---> [-1, 1]
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(2)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    # we decided on 200 steps for each episode
    for i in range(200):
        # observation
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            # GET THE ACTION
            movement = np.argmax(q_table[obs])
        else:
            movement = np.random.randint(0, 2)
        # Take the action!
        player.action(movement)
        enemy.f_d_action()
        food.f_d_action()


        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############
        # if the coordinates of the player equal the coordinates of the enemy,
        # negative reward
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        # if the coordinates of the player equal the coordinates of the food,
        # positive reward
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # first we need to obs immediately after the move, since q function requires two moves
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][movement]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][movement] = new_q
        # building the visual environment
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300), resample=Image.BOX)  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: # we create a display delay if sims ends
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'): # otherwise we keep going
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    # track improvements of system for a graph
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

