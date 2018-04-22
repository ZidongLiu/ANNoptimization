import numpy as np
import copy
import warnings
import scipy.optimize as sciopt
import warnings

class ANN:
    def __init__(self, net_strct, bias_strct,f = None, g = None):
        if len(bias_strct)+1!=len(net_strct):
            warnings.warn('The length of bias_strct and net_strct does not match!')
        self.net_strct = net_strct
        self.bias_strct = bias_strct
        self.layer = len(net_strct)
        self.train_error = None
        self.test_error = None
        self.step = None
        self.exit_flag = None
        self.output_scale = None
        self.input_scale = None
        self.weight_update = None
        self.weight = [None]*len(bias_strct)
        for i in range(0,len(bias_strct)):
            self.weight[i] = np.zeros((net_strct[i+1],net_strct[i]+bias_strct[i]))
        if f is None and g is None:
            def this_f(x):
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

            def this_g(x):
                return 1 - np.power(x,2)

            self.f = [this_f] * len(bias_strct)
            self.g = [this_g] * len(bias_strct)
        else:
            self.f = f
            self.g = g

    def BP_fit(self,train_X, train_Y,
           test_X = None, test_Y = None,
           learning_param = dict(alpha=0.01,  w_ini_ran=[-1, 1],
                                 max_step=100000,
                                 tol=0.05,scale_x=1,scale_y=0),
           monitor_param = dict(freq=5, monitor_err='rmse')):

        ### copy the dataset
        this_train_X = copy.deepcopy(train_X)
        this_train_Y = copy.deepcopy(train_Y)

        this_test_X = copy.deepcopy(test_X)
        this_test_Y = copy.deepcopy(test_Y)

        #### scaling to [-1,1]
        if learning_param['scale_x'] == 1:
            max_X = np.max(np.abs(this_train_X), 1)
            this_train_X = np.dot(np.diag(1 / max_X), this_train_X)

            if test_X is not None:
                this_test_X = np.dot(np.diag(1 / max_X), this_test_X)

        if learning_param['scale_y'] == 1:
            max_Y = np.max(np.abs(this_train_Y), 1)
            this_train_Y = np.dot(np.diag(1 / max_Y), this_train_Y)

            if test_Y is not None:
                this_test_Y = np.dot(np.diag(1 / max_Y), this_test_Y)

        ### initialize error set
        monit_ind = 0
        training_err = np.zeros((1, int(learning_param['max_step'] / monitor_param['freq'])))
        test_err = np.zeros(np.shape(training_err))


        ##### weight initialization
        w_ini = copy.deepcopy(self.weight)
        a = copy.deepcopy(learning_param['w_ini_ran'][0])
        b = copy.deepcopy(learning_param['w_ini_ran'][1])
        for i in range(len(self.weight)):
            w_ini[i] = (b - a) * np.random.rand(self.weight[i].shape[0], self.weight[i].shape[1]) + a

        ### bplearn: online learning
        monit_ind = 0
        exit_flag = 0
        w_old = w_ini
        print('Start leaning')

        for step in range(learning_param['max_step']):

            online_ind  = np.random.randint(0,this_train_X.shape[1])
            online_x = np.reshape(this_train_X[:,online_ind:(online_ind+1)],newshape=(this_train_X.shape[0],1))
            online_y = np.reshape(this_train_Y[:,online_ind:(online_ind+1)],newshape=(this_train_Y.shape[0],1))

            y_current = self.feedforward(w_old,online_x)
            delta_current = self.backprop(w_old,y_current,online_y)
            gradient_online = self.gradient(delta_current,y_current)

            ## weight update
            w_new = copy.deepcopy(w_old)
            for i in range(len(w_new)):
                if isinstance(learning_param['alpha'], (int, float)):
                    for i in range(len(w_ini)):
                        w_new[i] = w_old[i] - gradient_online[i] * learning_param['alpha']
                else:
                    for i in range(len(w_ini)):
                        w_new[i] = w_old[i] - gradient_online[i] * learning_param['alpha'](step + 1)

            w_old = w_new

            ### calculate training error
            this_train_pred = self.feedforward(w_old, this_train_X)[-1]

            if monitor_param['monitor_err'] == 'hit':
                this_train_pred = self.classify(this_train_pred)

            this_training_err = self.calculate_err(this_train_pred, this_train_Y, monitor_param['monitor_err'])

            if (step + 1) % monitor_param['freq'] == 0:
                ### update the training error
                training_err[0, monit_ind] = this_training_err

                ##### update test error if required
                if this_test_X is not None:
                    this_test_pred = self.feedforward(w_old, this_test_X)[-1]

                    if monitor_param['monitor_err'] == 'hit':
                        this_test_pred = self.classify(this_test_pred)

                    test_err[0, monit_ind] = self.calculate_err(this_test_pred, this_test_Y,
                                                                monitor_param['monitor_err'])

                monit_ind += 1

            ### judge if the exiting_criteria is reached
            if this_training_err < learning_param['tol']:
                exit_flag = 1
                break

        ### end of the learning process
        if learning_param['scale_x'] == 1:
            self.input_scale = copy.deepcopy(max_X)

        if learning_param['scale_y'] == 1:
            self.output_scale = copy.deepcopy(max_Y)

        self.exit_flag = exit_flag
        self.step = step
        self.train_error = copy.deepcopy(training_err)
        self.test_error = copy.deepcopy(test_err)
        self.weight = copy.deepcopy(w_old)

    def SA_fit(self, train_X, train_Y,
           test_X = None, test_Y = None,
           learning_param = dict(alpha=0.005, u=1, w_ini_ran=[-1, 1],
                                 max_step=100000,
                                 tol=0.05,scale_x=1,scale_y=0,
                                 T0=5, c=0.95, Redstep=5),
           monitor_param = dict(freq=5, monitor_err='rmse')):

        ### copy the dataset
        this_train_X = copy.deepcopy(train_X)
        this_train_Y = copy.deepcopy(train_Y)

        this_test_X = copy.deepcopy(test_X)
        this_test_Y = copy.deepcopy(test_Y)

        #### scaling to [-1,1]
        if learning_param['scale_x'] == 1:
            max_X = np.max(np.abs(this_train_X),1)
            this_train_X = np.dot(np.diag(1/max_X),this_train_X)

            if test_X is not None:
                this_test_X = np.dot(np.diag(1 / max_X),this_test_X)


        if learning_param['scale_y'] == 1:
            max_Y = np.max(np.abs(this_train_Y),1)
            this_train_Y = np.dot(np.diag(1/max_Y),this_train_Y)

            if test_Y is not None:
                this_test_Y = np.dot(np.diag(1/max_Y),this_test_Y)


        ### initialize error set
        monit_ind = 0
        training_err = np.zeros((1,int(learning_param['max_step']/monitor_param['freq'])))
        test_err = np.zeros(np.shape(training_err))

        ### T = T_0
        T = learning_param['T0']
        ##### weight initialization
        w_ini = copy.deepcopy(self.weight)
        a = copy.deepcopy(learning_param['w_ini_ran'][0])
        b = copy.deepcopy(learning_param['w_ini_ran'][1])
        for i in range(len(self.weight)):
            w_ini[i] = (b-a)*np.random.rand(self.weight[i].shape[0],self.weight[i].shape[1]) + a

        ##### learning process
        exiting_flag = 0
        w_old = copy.deepcopy(w_ini)
        for step in range(learning_param['max_step']):
            this_dir = self.getdir(learning_param['u'])
            w_new = copy.deepcopy(w_old)
            if isinstance(learning_param['alpha'],(int,float)):
                for i in range(len(w_ini)):
                    w_new[i] = w_old[i] + this_dir[i] * learning_param['alpha']
            else:
                for i in range(len(w_ini)):
                    w_new[i] = w_old[i] + this_dir[i] * learning_param['alpha'](step+1)

            #online_ind  = np.random.randint(0,this_train_X.shape[1])
            #online_x = this_train_X[:,online_ind:(online_ind+1)]
            #online_y = this_train_Y[:,online_ind:(online_ind+1)]

            pred_old = self.feedforward(w_old,this_train_X)[-1]
            pred_new = self.feedforward(w_new,this_train_X)[-1]
            loss_old = self.calculate_err(pred_old,this_train_Y,'rmse')
            loss_new = self.calculate_err(pred_new,this_train_Y,'rmse')

            if loss_old > loss_new:
                w_old = w_new
                #print('+   '+str(T))
                print('update' + '    step:'+str(step))
            else:
                this_p = np.exp((loss_old-loss_new)/T)
                #print('-   ' + str(T) + str(this_p))
                this_rand = np.random.rand(1)[0]
                if this_rand < this_p:
                    w_old = w_new
                    print('-   ' + str(T) + str(this_p))

            ### calculate training error
            this_train_pred = self.feedforward(w_old, this_train_X)[-1]

            if monitor_param['monitor_err'] == 'hit':
                this_train_pred = self.classify(this_train_pred)

            this_training_err = self.calculate_err(this_train_pred, this_train_Y, monitor_param['monitor_err'])

            if (step+1)%monitor_param['freq']==0:
                ### update the training error
                training_err[0,monit_ind] = this_training_err


                ##### update test error if required
                if this_test_X is not None:
                    this_test_pred = self.feedforward(w_old,this_test_X)[-1]

                    if monitor_param['monitor_err'] == 'hit':
                        this_test_pred = self.classify(this_test_pred)

                    test_err[0,monit_ind] = self.calculate_err(this_test_pred,this_test_Y,monitor_param['monitor_err'])

                monit_ind +=1

            ### judge if the exiting_criteria is reached
            if this_training_err < learning_param['tol']:
                exiting_flag = 1
                break

            ####
            #### update the temperature
            if (step+1)%learning_param['Redstep'] == 0:
                T = T*learning_param['c']
        ### end of the learning process
        if learning_param['scale_x'] ==1:
            self.input_scale = copy.deepcopy(max_X)

        if learning_param['scale_y'] == 1:
            self.output_scale = copy.deepcopy(max_Y)

        self.exit_flag = exiting_flag
        self.step = step
        self.train_error = copy.deepcopy(training_err)
        self.test_error = copy.deepcopy(test_err)
        self.weight = copy.deepcopy(w_new)


    def CG_fit(self, train_X, train_Y,
               test_X=None, test_Y=None,
               learning_param=dict(alpha=0.001, w_ini_ran=[-1, 1],
                                   n=10, max_online=10000,
                                   tol=1e-5, scale_x=1, scale_y=0),
               monitor_param=dict(freq=5, monitor_err='rmse')):
        ### raise the warning
        warnings.filterwarnings('error')
        learn_rate = learning_param['alpha']
        print(learn_rate)
        ### copy the dataset
        this_train_X = copy.deepcopy(train_X)
        this_train_Y = copy.deepcopy(train_Y)

        this_test_X = copy.deepcopy(test_X)
        this_test_Y = copy.deepcopy(test_Y)
        #### scaling to [-1,1]
        if learning_param['scale_x'] == 1:
            max_X = np.max(np.abs(this_train_X), 1)
            this_train_X = np.dot(np.diag(1 / max_X), this_train_X)

            if test_X is not None:
                this_test_X = np.dot(np.diag(1 / max_X), this_test_X)

        if learning_param['scale_y'] == 1:
            max_Y = np.max(np.abs(this_train_Y), 1)
            this_train_Y = np.dot(np.diag(1 / max_Y), this_train_Y)

            if test_Y is not None:
                this_test_Y = np.dot(np.diag(1 / max_Y), this_test_Y)

        ### initialize error set

        monit_ind = 0
        training_err = np.zeros((1, int(learning_param['max_online']*learning_param['n'] / monitor_param['freq'])))
        test_err = np.zeros(np.shape(training_err))

        ### initialize weight storing list

        stored_weight = [None]*int(learning_param['max_online']*learning_param['n'] / monitor_param['freq'])

        ##### weight initialization
        w_ini = copy.deepcopy(self.weight)
        a = copy.deepcopy(learning_param['w_ini_ran'][0])
        b = copy.deepcopy(learning_param['w_ini_ran'][1])
        for i in range(len(self.weight)):
            w_ini[i] = (b - a) * np.random.rand(self.weight[i].shape[0], self.weight[i].shape[1]) + a

        #### cg: online learning
        weight_update = [1]
        w_old = w_ini
        exit_flag = 0
        monit_ind = 0
        step = 0
        ### loop
        for i in range(learning_param['max_online']):
            print(i)
            online_ind = np.random.randint(0, this_train_X.shape[1])
            online_x = np.reshape(this_train_X[:, online_ind:(online_ind + 1)], newshape=(this_train_X.shape[0], 1))
            online_y = np.reshape(this_train_Y[:, online_ind:(online_ind + 1)], newshape=(this_train_Y.shape[0], 1))

            y_old = self.feedforward(w_old, online_x)
            delta_old = self.backprop(w_old, y_old, online_y)
            gradient_old = self.gradient(delta_old, y_old)
            gradient_old_col_vec = self.flat(gradient_old)

            d_array_col_vec = - gradient_old_col_vec
            d_array_nparray = np.asarray(d_array_col_vec)
            d_array_nparray.shape = (d_array_nparray.shape[0],)

            for j in range(learning_param['n']):
                ### line search
                print('linef     '+str(step))
                def line_f(w): ## this take a numpy array as input
                    w_line_f = np.asmatrix(w)
                    w_line_f.shape = (w_line_f.shape[1],1)
                    ### change vector of w into the ANN weight
                    w_weight = self.combine(w_line_f,self.weight)
                    y_line_f = self.feedforward(w_weight,online_x)[-1]
                    return self.calculate_err(y_line_f,online_y,'rmse')

                print('line_g     ' )
                def line_g(w):### w is a numpy array
                    ### transform nparray into matrix
                    w_line_g = np.asmatrix(w)
                    ### change the matrix into a col vector
                    w_line_g.shape = (w_line_g.shape[1],1)
                    ### change the col vector into the ANN weight
                    w_weight = self.combine(w_line_g,self.weight)
                    y_line_g = self.feedforward(w_weight,online_x)
                    delta_line_g = self.backprop(w_weight,y_line_g,online_y)
                    gra_line_g = self.gradient(delta_line_g,y_line_g)

                    ### change the shape of gradient into a col vector
                    gra_line_g = self.flat(gra_line_g)
                    ### change the vector into a array
                    gra_line_g = np.asarray(gra_line_g)
                    ### change the shape of the array
                    gra_line_g.shape = (gra_line_g.shape[0],)
                    ### change the array into list
                    gra_line_g = np.ndarray.tolist(gra_line_g)
                    ### return the lsit
                    return gra_line_g
                ### step 4 calculate training error
                this_train_pred = self.feedforward(w_old, this_train_X)[-1]

                if monitor_param['monitor_err'] == 'hit':
                    this_train_pred = self.classify(this_train_pred)

                this_training_err = self.calculate_err(this_train_pred, this_train_Y, monitor_param['monitor_err'])

                if (step + 1) % monitor_param['freq'] == 0:
                    ### update the training error
                    training_err[0, monit_ind] = this_training_err
                    stored_weight[monit_ind] = w_old
                    ##### update test error if required
                    if this_test_X is not None:
                        this_test_pred = self.feedforward(w_old, this_test_X)[-1]

                        if monitor_param['monitor_err'] == 'hit':
                            this_test_pred = self.classify(this_test_pred)

                        test_err[0, monit_ind] = self.calculate_err(this_test_pred, this_test_Y,
                                                                    monitor_param['monitor_err'])

                    monit_ind += 1

                step += 1

                w_old_col_vec = self.flat(w_old)
                w_old_nparray = np.asarray(w_old_col_vec)
                w_old_nparray.shape = (w_old_nparray.shape[0],) ### change into nparray

                try:
                    alpha = sciopt.line_search(line_f,line_g,xk=w_old_nparray,pk=d_array_nparray)[0]
                    weight_update += [1]
                except:
                    alpha = 0
                    weight_update += [0]
                    break


                print('alpha= ', alpha)

                ### weight update  step 3
                w_new_nparray = w_old_nparray + learn_rate*(alpha * d_array_nparray)
                w_old_nparray = copy.deepcopy(w_new_nparray)
                w_new_col_vec = w_old_col_vec + learn_rate*(alpha * d_array_col_vec)
                w_old_col_vec = copy.deepcopy(w_new_col_vec)
                w_old = self.combine(w_old_col_vec,self.weight)

                print(len(w_old))

                ### judge if the exiting_criteria is reached
                if this_training_err < learning_param['tol']:
                    exit_flag = 1
                    break

                ### step 5 and 6 update d and beta
                if j < learning_param['n']-1 :
                    ## get the new gradient
                    y_new = self.feedforward(w_old, online_x)
                    print('y_new')
                    delta_new = self.backprop(w_old, y_new, online_y)
                    gradient_new = self.gradient(delta_new, y_new)

                    gradient_new_col_vec = self.flat(gradient_new)

                    beta_nomi = np.dot(gradient_new_col_vec.T,(gradient_new_col_vec-gradient_old_col_vec))[0,0]
                    beta_deno = np.dot(gradient_old_col_vec.T,gradient_old_col_vec)[0,0]
                    beta = beta_nomi/beta_deno

                    d_array_col_vec = - gradient_new_col_vec + beta * d_array_col_vec
                    d_array_nparray = np.asarray(d_array_col_vec)
                    d_array_nparray.shape = (d_array_nparray.shape[0],)

            if exit_flag ==1:
                break

        ### end of the learning process
        if learning_param['scale_x'] == 1:
            self.input_scale = copy.deepcopy(max_X)

        if learning_param['scale_y'] == 1:
            self.output_scale = copy.deepcopy(max_Y)

        ### check if the ultimate weight is in the learning process
        min_err = np.argmin(training_err)
        self.exit_flag = exit_flag
        self.step = step
        self.train_error = copy.deepcopy(training_err)
        self.test_error = copy.deepcopy(test_err)
        self.weight = copy.deepcopy(stored_weight[min_err])
        self.weight_update = weight_update







    def getdir(self,u):
        dir = copy.deepcopy(self.weight)
        if isinstance(u,(int,float)):
            for i in range(len(dir)):
                dir[i] = u*(np.random.rand(dir[i].shape[0],dir[i].shape[1])-0.5)
        else:
            for i in range(len(dir)):
                dir[i] = np.multiply(u[i],np.random.rand(dir[i].shape[0],dir[i].shape[1])-0.5)

        return dir

    def feedforward(self,weight,x):
        this_y = [None]*self.layer
        #### the input should be turned into a columns wise form
        this_y[0] = np.append(np.ones((self.bias_strct[0],x.shape[1])),x,axis=0)
        for i in range(1,self.layer-1):
            inter = self.f[i-1](np.dot(weight[i-1],this_y[i-1]))
            this_y[i] = np.append(np.ones((self.bias_strct[i],inter.shape[1])),inter,axis=0)

        this_y[self.layer-1] = self.f[self.layer-2](np.dot(weight[self.layer-2],this_y[self.layer-2]))

        return this_y

    def backprop(self,weight,y,D):
        delta = [None]*(self.layer-1)
        delta[-1] = np.dot(np.diagflat(np.reshape(self.g[-1](y[-1]),newshape=(y[-1].shape[0],))), D-y[-1])
        for i in range(self.layer-3,-1,-1):
            temp = self.g[i](y[i+1][(self.bias_strct[i+1]):,0])
            delta[i] = np.dot(np.dot(np.diagflat(np.reshape(temp,newshape=temp.shape[0],)),np.transpose(weight[i+1][:,(self.bias_strct[i+1]):])),delta[i+1])

        return delta

    def gradient(self,delta,y):
        g_w = copy.deepcopy(self.weight)
        for i in range(self.layer-1):
            g_w[i] = - np.dot(np.reshape(delta[i],(delta[i].shape[0],1)),np.transpose(y[i]))

        return g_w

    def predict(self,x):
        if self.input_scale is not None:
            this_x = np.dot(np.diag(1/self.input_scale),x)
        else:
            this_x = copy.deepcopy(x)

        y_pred = self.feedforward(self.weight,this_x)[-1]

        if self.output_scale is not None:
            return np.dot(np.diag(self.output_scale),y_pred)
        else:
            return y_pred


    @staticmethod
    def classify(y):
        a = np.zeros(y.shape)
        max_ind = np.argmax(y,axis=0)
        for i in range(y.shape[1]):
            a[max_ind[i],i] = 1
        return a

    @staticmethod
    def calculate_err(y_pred, y_true, err_type):
        if err_type == 'mse':
            return np.mean(np.square(y_pred - y_true))
        elif err_type == 'rmse':
            return np.sqrt(np.mean(np.square(y_pred - y_true)))
        elif err_type == 'hit':
            ### return the misclassification rate, the lower the better
            dif = np.abs(y_pred - y_true)
            sum_dif = np.sum(dif,axis=0)
            sum_dif[sum_dif!=0]==1
            return np.mean(sum_dif)

    @staticmethod
    def flat(w):
        result = []
        for i in range(len(w)):
            vector = np.reshape(w[i], ((np.shape(w[i])[0] * np.shape(w[i])[1]), 1))
            result = np.append(result, vector)
        return np.matrix(result).T  # column matrix

    @staticmethod
    def combine(x, w):
        result = []
        index = 0
        for i in range(len(w)):
            shape = np.shape(w[i])
            lw = x[index:(index + (shape[0] * shape[1]))]
            index = index + (shape[0] * shape[1])
            lw_matrix = np.reshape(lw, (shape[0], shape[1]))
            result.append(lw_matrix)
        return result  # list of matrix

