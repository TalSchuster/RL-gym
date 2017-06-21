import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf
import sys


def buildAgentNet(observation, input_size, output_size):
    W1 = tf.get_variable("W1", initializer=tf.contrib.layers.xavier_initializer(), shape=[input_size,15], dtype=tf.float64)
    b1 = tf.get_variable("b1", initializer=tf.contrib.layers.xavier_initializer(), shape=[15], dtype=tf.float64)

    h1 = tf.tanh(tf.add(tf.matmul(observation, W1), b1))

    W2 = tf.get_variable("W2", initializer=tf.contrib.layers.xavier_initializer(), shape=[15,15], dtype=tf.float64)
    b2 = tf.get_variable("b2", initializer=tf.contrib.layers.xavier_initializer(), shape=[15], dtype=tf.float64)

    h2 = tf.tanh(tf.add(tf.matmul(h1, W2), b2))

    W3 = tf.get_variable("W3", initializer=tf.contrib.layers.xavier_initializer(), shape=[15,output_size], dtype=tf.float64)
    b3 = tf.get_variable("b3", initializer=tf.contrib.layers.xavier_initializer(), shape=[output_size], dtype=tf.float64)

    return tf.nn.softmax(tf.add(tf.matmul(h2, W3), b3))

def discount_rewards(r, gamma=0.99):
    discounted_rewards = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_rewards[t] = running_add
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)
    return discounted_rewards

class Agent():
    def __init__(self, actions_size, states_size, batch_size):
        self.observation = tf.placeholder(tf.float64, shape=[None,states_size])
        self.y = buildAgentNet(self.observation, states_size, actions_size)

        self.rewards_holder = tf.placeholder(shape=[None],dtype=tf.float64)
        self.actions_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        # Gets the indexes of the output parameters of the actions that were
        # taken in a batch or a single action for a single step.
        indexes = tf.range(0, tf.shape(self.y)[0]) * tf.shape(self.y)[1] + self.actions_holder
        ys_of_taken_actions = tf.gather(tf.reshape(self.y, [-1]), indexes)

        # Note that the gradients function of tensorflow sums  over all rows ,so
        # the gradients will be sum over all taken actions.
        loss = - (tf.log(ys_of_taken_actions) * self.rewards_holder) / batch_size

        self.train_vars = tf.trainable_variables()
        self.gradients = tf.gradients(loss,self.train_vars)

        # For assigning loaded parameters to the model
        self.parameters_assign = []
        self.parameters_W = tf.placeholder(tf.float64, shape=[None,None])
        self.parameters_b = tf.placeholder(tf.float64, shape=[None])
        for i,var in enumerate(self.train_vars):
            if i % 2 ==0:
                self.parameters_assign.append(var.assign(self.parameters_W))
            else:
                self.parameters_assign.append(var.assign(self.parameters_b))

        # alternative: regular tf saver
        self.saver = tf.train.Saver()

        # For updating manually the gradients after a full batch
        self.gradient_holders = []
        for i, _ in enumerate(self.train_vars):
            grad_holder = tf.placeholder(tf.float64,name=str(i)+'_holder')
            self.gradient_holders.append(grad_holder)


        adam = tf.train.AdamOptimizer(learning_rate=0.005)
        self.train_op = adam.apply_gradients(zip(self.gradient_holders,self.train_vars))


def main(argv):
    tf.reset_default_graph()
    total_episodes = 30000
    batch_size = 10
    load_parameters = False
    save_parameters = True
    render = False
    env_d = 'LunarLander-v2'
    if len(argv) > 1:
        env_d = argv[1]

    agent = None
    if env_d == 'LunarLander-v2':
        agent = Agent(4, 8, batch_size)
    elif env_d == 'CartPole-v0':
        agent = Agent(2, 4, batch_size)
    else:
        print('Not supported enviroment: ' + env_d)
        sys.exit(1)

    env = gym.make(env_d)
    init = tf.global_variables_initializer()
    best_rewards = -np.inf
    with tf.Session() as sess:
        sess.run(init)
        obsrv = env.reset() # Obtain an initial observation of the environment
        episode_number = 0

        if load_parameters:
            fd = 'ws_' + env_d + '.p.best'
            params = pickle.load(open(fd,'r'))
            for i,var in enumerate(params):
                if i % 2 ==0:
                    sess.run(agent.parameters_assign[i], feed_dict={agent.parameters_W:var})
                else:
                    sess.run(agent.parameters_assign[i], feed_dict={agent.parameters_b:var})

        grad_buffer = sess.run(agent.train_vars)
        for i, grad in enumerate(grad_buffer):
            grad_buffer[i] = grad * 0

        all_rewards = []
        while episode_number < total_episodes:
            done = False
            game_states = []
            game_actions = []
            game_rewards = []
            while not done:
                if render:
                    env.render()
                game_states.append(obsrv)
                # Run the policy network and get a distribution over actions
                action_probs = sess.run(agent.y, feed_dict={agent.observation: [obsrv]})

                # sample action from distribution
                action = np.argmax(np.random.multinomial(1,action_probs[0,]))
                game_actions.append(action)

                # step the environment and get new measurements
                obsrv, reward, done, info = env.step(action)
                game_rewards.append(reward)

            game_states = np.array(game_states)
            game_actions = np.array(game_actions)
            game_rewards = np.array(game_rewards)

            all_rewards.append(np.sum(game_rewards))

            # sum gradients of the game (divided by batch_size)
            feed_dict={agent.rewards_holder:discount_rewards(game_rewards),
                agent.actions_holder:game_actions,agent.observation:np.vstack(game_states)}
            game_grads = sess.run(agent.gradients, feed_dict=feed_dict)
            for i, grad in enumerate(game_grads):
                grad_buffer[i] += grad

            if episode_number % batch_size == 0 and episode_number != 0:
                # apply the batch gradients on the model
                feed_dict = dict(zip(agent.gradient_holders, grad_buffer))
                sess.run(agent.train_op, feed_dict=feed_dict)
                for i, grad in enumerate(grad_buffer):
                    grad_buffer[i] = grad * 0

            if episode_number % 100 == 0 and episode_number != 0:
                rewards_mean = np.mean(all_rewards[-100:])
                print('games: %i, reward mean of last 100: %.2f' % \
                (episode_number, rewards_mean ))
                if rewards_mean >= best_rewards and save_parameters:
                    best_rewards = rewards_mean
                    params = sess.run(agent.train_vars)
                    fd = 'ws_' + env_d + '.p.best'
                    pickle.dump(params, open(fd,'wb'))

            if episode_number % 1000 == 0 and save_parameters:
                agent.saver.save(sess, 'model_' + env_d + '.ckpt')

            episode_number += 1
            obsrv = env.reset()

        if save_parameters:
            params = sess.run(agent.train_vars)
            fd = 'ws_' + env_d + '.p'
            pickle.dump(params, open(fd,'wb'))




if __name__ == '__main__':
    tf.app.run()
