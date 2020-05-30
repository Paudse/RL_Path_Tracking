import os
import gym
import numpy as np
import pandas as pd
import tensorflow as tf

class PPO:
    def __init__(self, batch, t='ppo2'):
        self.t = t
        # self.ep = ep
        self.batch = batch
        self.log = 'model/{}_log'.format(t)
        self.wlists =[]

        # self.bound_st = 1 # steering boundary
        # self.bound_dv_h = 1# acceleration boundary
        # self.bound_dv_l = -1
        self.bound = 20/180*np.pi

        self.gamma = 1 # 0.99
        self.A_LR = 0.001
        self.C_LR = 0.002
        self.A_UPDATE_STEPS = 10 #10
        self.C_UPDATE_STEPS = 10 #10
        self.S_DIM = 2
        self.A_DIM = 1

        # KL penalty, d_target、β for ppo1
        # self.kl_target = 0.01
        # self.lam = 0.5
        # ε for ppo2
        self.epsilon = 0.05

        self.sess = tf.Session()
        # self.sess = tf.InteractiveSession()
        self.build_model()
        self.saver=tf.train.Saver(max_to_keep=5)

        try:
            model_iteration_cycle_file = tf.train.latest_checkpoint('check_point/')
            self.saver.restore(self.sess,model_iteration_cycle_file)
            print('restore previous NN data')
            self.get_weights()         
        except:
            print('no previous NN data, start new')

    def load_weights(self):
        try:
            model_iteration_cycle_file = tf.train.latest_checkpoint('check_point/')
            self.saver.restore(self.sess,model_iteration_cycle_file)
            # print('restore previous NN data')
            self.get_weights()         
        except:
            print('no previous NN data, start new')

    def _build_critic(self):
        """critic model.
        """
        with tf.variable_scope('critic'):
            x = tf.layers.dense(self.states, 20, tf.nn.relu)
            x = tf.layers.dense(x, 20, tf.nn.relu)
            # x = tf.layers.dense(x, 3, tf.nn.relu)
            # x = tf.layers.dense(self.states, 10, tf.nn.relu)
            # x = tf.layers.dense(self.states, 5, tf.nn.relu)
            # x = tf.layers.dense(self.states, 4, tf.nn.relu)
            # x = tf.layers.dense(self.states, 3, tf.nn.relu)
            # x = tf.layers.dense(self.states, 2, tf.nn.relu)

            self.v = tf.layers.dense(x, 1)
            self.advantage = self.dr - self.v

    def _build_actor(self, name, trainable):
        """actor model.
        """
        with tf.variable_scope(name):
            x = tf.layers.dense(self.states, 20, tf.nn.relu, trainable=trainable)
            x = tf.layers.dense(x, 20, tf.nn.relu, trainable=trainable)
            # x = tf.layers.dense(x, 3, tf.nn.relu, trainable=trainable)
            # x = tf.layers.dense(self.states, 10, tf.nn.relu, trainable=trainable)
            # x = tf.layers.dense(self.states, 6, tf.nn.relu, trainable=trainable)
            # x = tf.layers.dense(self.states, 5, tf.nn.relu, trainable=trainable)
            # x = tf.layers.dense(self.states, 4, tf.nn.relu)
            # x = tf.layers.dense(self.states, 3, tf.nn.relu)
            # x = tf.layers.dense(self.states, 2, tf.nn.relu)
            # x = tf.layers.dense(self.states, 20, tf.nn.relu, trainable=trainable)
            # x = tf.layers.dense(self.states, 10, tf.nn.relu, trainable=trainable)
            # x = tf.layers.dense(self.states, 5, tf.nn.relu)

            mu = self.bound * tf.layers.dense(x, self.A_DIM, tf.nn.tanh, trainable=trainable)
            # mu = 0 * tf.layers.dense(x, self.A_DIM, tf.nn.tanh, trainable=trainable)
            # print('mu=',mu)
            # sigma = tf.layers.dense(x, self.A_DIM, tf.nn.softplus, trainable=trainable)
            sigma = tf.layers.dense(x, self.A_DIM, tf.nn.softplus, trainable=trainable)
            # sigma = 0.01*tf.layers.dense(x, self.A_DIM, tf.nn.softmax, trainable=trainable)
            # sigma = 0.01*tf.layers.dense(x, self.A_DIM, 1, trainable=trainable)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            # print(norm_dist)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        # print('prarms=', params)
        # self.sess.run(params)

        return norm_dist, params

    def build_model(self):
        """build model with ppo loss.
        """
        # inputs
        self.states = tf.placeholder(tf.float32, [None, self.S_DIM], 'states')
        self.action = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        # print(self.action)
        self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.dr = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        # build model
        self._build_critic()
        nd, pi_params = self._build_actor('actor', trainable=True)
        old_nd, oldpi_params = self._build_actor('old_actor', trainable=False)

        # define ppo loss
        with tf.variable_scope('loss'):
            # critic loss
            self.closs = tf.reduce_mean(tf.square(self.advantage))

            # actor loss
            with tf.variable_scope('surrogate'):
                ratio = tf.exp(nd.log_prob(self.action) - old_nd.log_prob(self.action))
                # print('ratio =',ratio)
                surr = ratio * self.adv

            if self.t == 'ppo1':
                passs
                # self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                # kl = tf.distributions.kl_divergence(old_nd, nd)
                # self.kl_mean = tf.reduce_mean(kl)
                # self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else: 
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.- self.epsilon, 1.+ self.epsilon) * self.adv))

        # define Optimizer
        with tf.variable_scope('optimize'):
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(nd.sample(1), axis=0)
            # print('nd.sample(1)',nd.sample(1))

        # update old actor
        with tf.variable_scope('update_old_actor'):
            self.update_old_actor = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # tf.summary.FileWriter(self.log, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        # tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
        # self.sess.run(tf.constant_initializer(0))

        # print(self.adv .eval(session=self.sess))

        abc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # abc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print('abc=', abc)

        # print(self.sess.run(self.adv))


    def choose_action(self, state):
        """choice continuous action from normal distributions.

        Arguments:
            state: state.

        Returns:
           action.
        """
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.states: state})[0]
        # print('self.sample_op',self.sample_op)
        # print(self.sample_op,state,action)
        # action_dv = np.clip(action[0], self.bound_dv_l, self.bound_dv_h)
        action_st = np.clip(action, -self.bound, self.bound)

        return action_st

    def get_value(self, state):
        """get q value.

        Arguments:
            state: state.

        Returns:
           q_value.
        """
        # print('state1',state)
        if state.ndim < 2: state = state[np.newaxis, :]
        # print(state)

        return self.sess.run(self.v, {self.states: state})

    def discount_reward(self, states, rewards, next_observation):
        """Compute target value.

        Arguments:
            states: state in episode.
            rewards: reward in episode.
            next_observation: state of last action.

        Returns:
            targets: q targets.
        """
        s = np.vstack([states, next_observation.reshape(-1, self.S_DIM)])
        # print('s',s)
        # print('self.get_value(s)',self.get_value(s))
        q_values = self.get_value(s).flatten()
        # print('q_values',q_values)
        # print('rewards',rewards)

        targets = rewards + self.gamma * q_values[1:]
        # print('targets1',targets)
        targets = targets.reshape(-1, 1)
        # print('targets',targets)

        return targets

# not work.
#    def neglogp(self, mean, std, x):
#        """Gaussian likelihood
#        """
#        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
#               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
#               + tf.reduce_sum(tf.log(std), axis=-1)

    def update(self, states, action, dr, j):
        """update model.

        Arguments:
            states: states.
            action: action of states.
            dr: discount reward of action.
        """
        self.sess.run(self.update_old_actor)

        adv = self.sess.run(self.advantage,
                            {self.states: states,
                             self.dr: dr})

        # print(states)


        # update actor
        for _ in range(self.A_UPDATE_STEPS):
            # print(_,states)
            self.sess.run(self.atrain_op,
                          {self.states: states,
                           self.action: action,
                           self.adv: adv})
            # print(k)

        # update critic
        for _ in range(self.C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op,
                          {self.states: states,
                           self.dr: dr})
            # print(dr)


    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')


    def save_learning(self):
        self.saver.save(self.sess,'check_point/ppo.ckpt')
        tf.global_variables()
        # self.save_weights_csv()
        self.get_weights()

    def save_weights_csv(self):
        from tensorflow.python import pywrap_tensorflow
        import os

        checkpoint_path = os.path.join("check_point/ppo.ckpt")
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor_name: ", key)
            print(reader.get_tensor(key))

    def get_weights(self):
        # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(variables[0])
        # print('len(variables)',len(variables))
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        values = [self.sess.run(v) for v in variables]
        # print(values[0])
        lists=[]

        for i in range (len(variables)):
            lists.append(values[i].tolist())

        self.wlists.append(lists)

        # print(self.wlists)
        wb=pd.DataFrame(columns=variables,data=self.wlists)
        wb.to_csv('check_point/wbcsv.csv',encoding='gbk')

