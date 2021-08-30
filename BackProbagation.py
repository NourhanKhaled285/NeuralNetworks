import numpy as np
import  random
import math
class BackProbagation:
    def __init__(self,dataset,N_OF_hidden_layers, N_OF_Neurons,L_rate,N_Epochs,ActivationType,Bias):
        self.N_Classes=3
        self.N_features=dataset.shape[1]
        self.dataset=dataset
        self.N_OF_Neurons=N_OF_Neurons
        self.N_OF_hidden_layers=N_OF_hidden_layers
        self.L_rate=L_rate
        self.N_Epochs=N_Epochs
        self.ActivationType=ActivationType
        self.Bias=Bias
        self.total_weights_hidden = []
        self.weights_of_output=[]
        weights_of_layer = []
        for i in range(int(self.N_OF_Neurons[0])):
            weights_of_layer.append(np.random.rand(self.N_features + 1))
        self.total_weights_hidden.append(weights_of_layer)

        for i in range(self.N_OF_hidden_layers):
            weights_of_layer = []
            if (i + 1 < self.N_OF_hidden_layers):
                for j in range(int(self.N_OF_Neurons[i + 1])):
                    weights_of_layer.append(np.random.rand(int(self.N_OF_Neurons[i]) + 1))
                self.total_weights_hidden.append(weights_of_layer)

        for i in range(self.N_Classes):
            self.weights_of_output.append(np.random.rand(int(self.N_OF_Neurons[self.N_OF_hidden_layers - 1]) + 1))


    def split_TrainTest(self):

       np.random.shuffle(self.dataset[:50, :])
       np.random.shuffle(self.dataset[50:100, :])
       np.random.shuffle(self.dataset[100:150, :])

       train=np.concatenate((self.dataset[:30,:],self.dataset[50:80,:],self.dataset[100:130, :]),axis=0)
       test=np.concatenate((self.dataset[30:50,:],self.dataset[80:100,:],self.dataset[130:150,:]),axis=0)
       # print(len(train), test)
       return train,test


    def create_lables(self):
        t=np.empty([self.dataset.shape[0],1])
        #t = [None]*self.dataset.shape[0]
        for i in range (self.dataset.shape[0]):
            if i<50:
                t[i] =1

            elif i>=50 and i<100:
                t[i] = 2
            elif i >= 100 and i < 150:
                t[i] = 3
        t_train = np.concatenate((t[:30, :], t[50:80, :], t[100:130, :]), axis=0)
        t_test = np.concatenate((t[30:50, :], t[80:100, :],t[130:150, :]), axis=0)

        return t_train,t_test
        # print(t)

    def segmoid(self,net):
      f=1/(1+math.exp(-1*net))
      return f



    def tanh (self,net):
        f=(1-math.exp(-2*net))/(1+math.exp(-2*net))
        return f



    def feedforward(self,input):

        bias=[0]
        if (self.Bias) == 1:
            bias = [1]

        new_input=np.empty([1,int(self.N_OF_Neurons[0])])
        total_act_val_layers=[]
        layer_act_val=[]

        # print(input[0,:])
        concat = np.concatenate((bias, input), axis=0)
        x = np.array(concat)
        x = x.reshape(len(x), 1)
        # print('input = ',concat)

        for j in range(int(self.N_OF_Neurons[0])):

              net = np.dot(self.total_weights_hidden[0][j], x)[0]
              if self.ActivationType=='Sigmoid':
                  f=self.segmoid(net)
              else:
                  f=self.tanh(net)
              new_input[0,j]=f
        # print("first activation value = ",new_input)
        layer_act_val.append(new_input)
        total_act_val_layers.append(layer_act_val)
        final_hiddenlayer = new_input
        if self.N_OF_hidden_layers>1:
            i=1
            # print('number of hidden layers = ',self.N_OF_hidden_layers)
            while i <self.N_OF_hidden_layers:

                    layer_act_val=[]
                    concat = np.concatenate((bias, new_input[0,:]), axis=0)
                    x = np.array(concat)
                    x = x.reshape(len(x), 1)
                    # print('input = ', concat)
                    # print('x',x)
                    new_input = np.empty([1, int(self.N_OF_Neurons[i])])
                    for j in range(int(self.N_OF_Neurons[i])):
                        # print('weight xxx = ',self.total_weights_hidden[i][j])
                        net = np.dot(self.total_weights_hidden[i][j],x)[0]
                        if self.ActivationType == 'Sigmoid':
                            f = self.segmoid(net)
                        else:
                            f = self.tanh(net)
                        new_input[0, j] = f
                    layer_act_val.append(new_input)
                    final_hiddenlayer=new_input
                    i+=1
                    total_act_val_layers.append(layer_act_val)
        #print(final_hiddenlayer)
        # print(x)
        #print(self.total_weights_hidden[0][int(self.N_OF_Neurons[0])-1])

        # print(new_input)
        y=np.empty([1,self.N_Classes])
        for i in range(self.N_Classes):
            concat = np.concatenate((bias, final_hiddenlayer[0,:]), axis=0)
            x = np.array(concat)
            x = x.reshape(len(x), 1)
            net = np.dot(self.weights_of_output[i], x)[0]
            if self.ActivationType == 'Sigmoid':
                f = self.segmoid(net)
                y[0,i] = f
            else:
                f = self.tanh(net)
                y[0,i]=f
        # print("total net = \n",total_act_val_layers)
        # print("first layer net = \n", total_act_val_layers[0][0][0][0])
        return y,total_act_val_layers
        # print(self.weights_of_output)


    def calculate_error(self,t,y):
        error=0
        print(t)
        if t==1:
           print('DONE ***')
           error+=1-y[0,0]+0-y[0,1]+0-y[0,2]
        elif t==2:
            error += 0- y[0,0] + 1 - y[0,1] + 0 - y[0,2]
        elif t==3:
            error += 0- y[0,0] + 0 - y[0,1] + 1- y[0,2]
        return error





    def Backpropagate(self, y, target, total_act_val_layers):

        print('total hidden weights : ',self.total_weights_hidden)
        print('output weights : ', self.weights_of_output)
        total_gradient = []

        def collect_each_neuron_weights():
            weights_of_alllayers = []

            for layer in range(self.N_OF_hidden_layers -1):
                weights_of_layer = []
                for j in range(int(self.N_OF_Neurons[layer])):
                    weights_hidden_neuron = np.array([])
                    for i in range(int(self.N_OF_Neurons[layer+1])):
                        weights_hidden_neuron = np.append(weights_hidden_neuron, self.total_weights_hidden[layer+1][i][j + 1])
                    weights_of_layer.append(weights_hidden_neuron)
                weights_of_alllayers.append(weights_of_layer)



            weights_of_layer=[]
            for j in range(int(self.N_OF_Neurons[self.N_OF_hidden_layers-1])):
                      weights_hidden_neuron = np.array([])
                      for i in range(self.N_Classes):
                        weights_hidden_neuron=np.append(weights_hidden_neuron, self.weights_of_output[i][j+1])
                      weights_of_layer.append(weights_hidden_neuron)
            # print('hidden layer number {}'.format(layer+1),weights_of_layer)
            weights_of_alllayers.append(weights_of_layer)


            return weights_of_alllayers

        print('weights of each neural in each layer = \n', collect_each_neuron_weights())

        def calculate_total_gradient():
            gradient_outputlayer = np.array([])

            error = self.calculate_error(target, y)

            for i in range(self.N_Classes):

                # for j in range(int(self.N_OF_Neurons[self.N_OF_hidden_layers-1])):
                if self.ActivationType == 'Sigmoid':
                    # print(y[0,i])
                    # print('gradient : ', error * y[0, i] * (1 - y[0, i]))
                   gradient_outputlayer = np.append(gradient_outputlayer, error * y[0, i] * (1 - y[0, i]))
                else:
                    gradient_outputlayer = np.append(gradient_outputlayer, error * (1 - (y[0, i] ** 2)))


            #calculate gradient of hidden layers

            weights_of_alllayers=collect_each_neuron_weights()

            gradient_hidden_neuron = np.array([])
            print('gradient of output layer = ',gradient_outputlayer)
            for i in range(int(self.N_OF_Neurons[self.N_OF_hidden_layers - 1])):
                res = np.dot(weights_of_alllayers[self.N_OF_hidden_layers - 1][i], gradient_outputlayer.T)
                #print('res : ',res)
                res2 = res * total_act_val_layers[self.N_OF_hidden_layers - 1][0][0][i] * (1 - total_act_val_layers[self.N_OF_hidden_layers - 1][0][0][i])
                #print('res2: ', res2)
                gradient_hidden_neuron = np.append(gradient_hidden_neuron, res2)

            total_gradient.append(gradient_hidden_neuron)
            # print('total gradient = \n',self.total_gradient)




            layer=self.N_OF_hidden_layers-2
            # print('layer num = ',layer)
            layer_idx=0
            while layer>=0:
                 gradient_hidden_neuron = np.array([])
                 # print('grad of previous layer = ', self.total_gradient[layer_idx])
                 for i in range(int(self.N_OF_Neurons[layer])):
                     # print('weights_of_alllayers = ',weights_of_alllayers[layer][i])
                     res = np.dot(weights_of_alllayers[layer][i] ,total_gradient[layer_idx].T)
                     # print('res : ',res)
                     res2 = res * total_act_val_layers[layer][0][0][i] * ( 1 - total_act_val_layers[layer][0][0][i])
                     # print('res2: ',res2)
                     gradient_hidden_neuron = np.append(gradient_hidden_neuron, res2)
                 total_gradient.append(gradient_hidden_neuron)
                 #self.total_gradient.append(gradient_OF_layer)
                 layer-=1
                 layer_idx+=1
            return total_gradient, total_gradient[::-1],gradient_outputlayer




        # print('total weights for each neuron in each layer = \n', collect_each_neuron_weights())
        t_g,t_g_r,g_otput_l=calculate_total_gradient()
        print('total gradient = \n',t_g,'\ntotal gradient reversed = \n',t_g_r)
        print('output_layer gradient = \n',g_otput_l)
        return t_g,t_g_r,g_otput_l
        # calculate_total_gradient()
        # print(self.total_weights_hidden)
        # print(gradient_OF_layer)




    def update_weights(self,total_gradient,output_gradient):

       for i in range(self.N_OF_hidden_layers):
           for j in range(int(self.N_OF_Neurons[i])):
               self.total_weights_hidden[i][j]+=total_gradient[i][j]*self.L_rate

       for i in range(self.N_Classes):
           self.weights_of_output[i]+=output_gradient[i]*self.L_rate

       return self.total_weights_hidden,self.weights_of_output



    def training(self,train,t_train):
      # print(train.shape[0])
      for epoch in range(self.N_Epochs):
        for i in range(train.shape[0]):
          y,total_act_val_layers=self.feedforward(train[i, :])
          #print(y)
          total_gradient2,total_gradient,gradient_outputlayer=self.Backpropagate(y,t_train[i,0],total_act_val_layers)
          print(total_gradient2)
          total_weights_hidden,weights_of_output=self.update_weights(total_gradient,gradient_outputlayer)
          print('\nhidden weights updated = \n',total_weights_hidden,'\n')
          print('output layer weights updated = \n', weights_of_output)
          print('*****************************************************************************************\n')

    def testing(self,test,t_test):


        conf_matrix = [[0 for i in range(3)],[0 for i in range(3)],[0 for i in range(3)]]

        print(conf_matrix)
        # print(conf_matrix[0][0])
        # print(test.shape[0])
        for i in range (test.shape[0]):
            y,total_act=self.feedforward(test[i,:])
            max_neuron=max(y[0,1], y[0,0],y[0,2])
            # print('max = ',max_neuron)
            # print('target is = ',t_test[i,0])
            if t_test[i,0] == 1:

                if max_neuron==y[0,0]:

                    conf_matrix[0][0]+=1
                elif max_neuron==y[0,1]:

                    conf_matrix[0][1]+=1
                elif max_neuron==y[0,2] :

                    conf_matrix[0][2]+=1

            elif t_test[i,0] == 2:
                if max_neuron==y[0,0]:
                    conf_matrix[1][0]+=1

                elif max_neuron == y[0, 1]:
                    conf_matrix[1][1]+=1

                elif max_neuron == y[0, 2]:
                    conf_matrix[1][2]+=1

            elif t_test[i,0] == 3:
                if max_neuron == y[0, 0]:
                  conf_matrix[2][0]+=1
                elif max_neuron == y[0, 1]:
                    conf_matrix[2][1]+=1
                elif max_neuron == y[0, 2]:
                    conf_matrix[2][2]+=1
        # print("conf list = \n",conf_matrix)
        return np.array(conf_matrix)





    def classify(self):
        train,test=self.split_TrainTest()
        t_train, t_test = self.create_lables()
        self.training(train,t_train)
        print('original hidden weighs = \n',self.total_weights_hidden)
        print('original output weighs = \n', self.weights_of_output)
        Confusion_Matrix=self.testing(test,t_test)
        print('conf matrix is = \n', Confusion_Matrix)
        print("accuracy is = ", ((Confusion_Matrix[0][0] + Confusion_Matrix[1][1] + Confusion_Matrix[2][2]) / len(test)) * 100, "%")


