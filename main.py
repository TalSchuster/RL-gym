import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf

#env_d = 'LunarLander-v2'
env_d = 'CartPole-v0'
env = gym.make(env_d)
env.reset()


def agentLunar(observation):
    x = tf.expand_dims(observation,0)
    W1 = tf.Variable(tf.zeros([8,15]))
    b1 = tf.Variable(tf.zeros([15]))

    h1 = tf.tanh(tf.add(tf.matmul(x, W1), b1))

    W2 = tf.Variable(tf.zeros([15,15]))
    b2 = tf.Variable(tf.zeros([15]))

    h2 = tf.tanh(tf.add(tf.matmul(h1, W2), b2))

    W3 = tf.Variable(tf.zeros([15,4]))
    b3 = tf.Variable(tf.zeros([4]))

    return tf.nn.softmax(tf.add(tf.matmul(h2, W3), b3))


def agentCart(observation):
    x = tf.expand_dims(observation,0)
    W1 = tf.Variable(tf.zeros([4,15]))
    b1 = tf.Variable(tf.zeros([15]))

    h1 = tf.tanh(tf.add(tf.matmul(x, W1), b1))

    W2 = tf.Variable(tf.zeros([15,15]))
    b2 = tf.Variable(tf.zeros([15]))

    h2 = tf.tanh(tf.add(tf.matmul(h1, W2), b2))

    W3 = tf.Variable(tf.zeros([15,2]))
    b3 = tf.Variable(tf.zeros([2]))

    return tf.nn.softmax(tf.add(tf.matmul(h2, W3), b3))

if env_d == 'LunarLander-v2':
    observation = tf.placeholder(tf.float32, shape=[8])
    y = agentLunar(observation)

if env_d == 'CartPole-v0':
    observation = tf.placeholder(tf.float32, shape=[4])
    y = agentCart(observation)

adam = tf.train.AdamOptimizer()
init = tf.global_variables_initializer()
def main(argv):
    total_episodes = 30000
    batch_size = 10
    with tf.Session() as sess:
        sess.run(init)
        obsrv = env.reset() # Obtain an initial observation of the environment
        episode_number = 0
        ep_rewards = np.array([])
        ep_policies = np.array([])
        while episode_number <= total_episodes/ batch_size:
            batch_grads = np.zeros([])
            for b in xrange(batch_size):
                done = False
                while not done:
                    # Run the policy network and get a distribution over actions
                    action_probs = sess.run(y,feed_dict={observation: obsrv})
                    # sample action from distribution
                    action = np.argmax(np.random.multinomial(1,action_probs[:, 0]))
                    # step the environment and get new measurements
                    obsrv, reward, done, info = env.step(action)

                    ep_rewards = np.append(ep_rewards, reward)
                    ep_policies = np.append(ep_policies, action_probs)

                episode_number += 1
                obsrv = env.reset()

                sum_grads = np.zeros([])
                for t in xrange(len(ep_rewards)):
                    phrase = tf.add(tf.log(ep_policies[t]), tf.constant(np.sum(ep_rewards[t:])))
                    grads = adam.compute_gradients(phrase)
                    grad_placeholder = [(tf.placeholder("float", shape = grad[1].get_shape()), grad[1]) for grad in grads]
                    import ipdb; ipdb.set_trace()
                    print 'a'

                batch_grads += sum_grads

            batch_grads  = batch_grads/ batch_size
            adam.compute_gradients(batch_grads)
            sess.run(adam)


if __name__ == '__main__':
    tf.app.run()
