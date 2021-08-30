import numpy as np
import  random
import math
class BackProbagation_mnist:
    def __init__(self,train,test,target_train,target_test,N_OF_hidden_layers, N_OF_Neurons,L_rate,N_Epochs,ActivationType,Bias):
        self.N_Classes=10
        self.N_features=train.shape[1]
        self.train=train
        self.test=test
        self.target_train=target_train
        self.target_test=target_test
        self.N_OF_Neurons=N_OF_Neurons
        self.N_OF_hidden_layers=N_OF_hidden_layers
        self.L_rate=L_rate
        self.N_Epochs=N_Epochs
        self.ActivationType=ActivationType
        self.Bias=Bias
        self.total_weights_hidden = []
        self.weights_of_output=[]
        # print(self.N_features)
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

    def segmoid(self,net):
      f=1/(1+math.exp(-1*net))
      return f



    def tanh (self,net):
        f=(1-math.exp(-2*net))/(1+math.exp(-2*net))
        return f

    def feedforward(self, input):

        bias = [0]
        if (self.Bias) == 1:
            bias = [1]

        new_input = np.empty([1, int(self.N_OF_Neurons[0])])
        total_act_val_layers = []
        layer_act_val = []

        # print(input[0,:])
        concat = np.concatenate((bias, input), axis=0)
        x = np.array(concat)
        x = x.reshape(len(x), 1)

        for j in range(int(self.N_OF_Neurons[0])):

            net = np.dot(self.total_weights_hidden[0][j], x)[0]
            if self.ActivationType == 'Sigmoid':
                f = self.segmoid(net)
            else:
                f = self.tanh(net)
            new_input[0, j] = f
        # print("first activation value = ",new_input)
        layer_act_val.append(new_input)
        total_act_val_layers.append(layer_act_val)
        final_hiddenlayer = new_input
        if self.N_OF_hidden_layers > 1:
            i = 1
            # print('number of hidden layers = ',self.N_OF_hidden_layers)
            while i < self.N_OF_hidden_layers:

                layer_act_val = []
                concat = np.concatenate((bias, new_input[0, :]), axis=0)
                x = np.array(concat)
                x = x.reshape(len(x), 1)
                # print('x',x)
                new_input = np.empty([1, int(self.N_OF_Neurons[i])])
                for j in range(int(self.N_OF_Neurons[i])):
                    # print('weight xxx = ',self.total_weights_hidden[i][j])
                    net = np.dot(self.total_weights_hidden[i][j], x)[0]
                    if self.ActivationType == 'Sigmoid':
                        f = self.segmoid(net)
                    else:
                        f = self.tanh(net)
                    new_input[0, j] = f
                layer_act_val.append(new_input)
                final_hiddenlayer = new_input
                i += 1
                total_act_val_layers.append(layer_act_val)
        # print(final_hiddenlayer)
        # print(x)
        # print(self.total_weights_hidden[0][int(self.N_OF_Neurons[0])-1])

        # print(new_input)
        y = np.empty([1, self.N_Classes])
        for i in range(self.N_Classes):
            concat = np.concatenate((bias, final_hiddenlayer[0, :]), axis=0)
            x = np.array(concat)
            x = x.reshape(len(x), 1)
            net = np.dot(self.weights_of_output[i], x)[0]
            if self.ActivationType == 'Sigmoid':
                f = self.segmoid(net)
                y[0, i] = f
            else:
                f = self.tanh(net)
                y[0, i] = f
        # print("total net = \n",total_act_val_layers)
        # print("first layer net = \n", total_act_val_layers[0][0][0][0])
        return y, total_act_val_layers
        # print(self.weights_of_output)


    def calculate_error(self,t,y):
        error=0
        print(t)
        if t==0:
           print('DONE ***')
           error+= 1 - y[0,0] - y[0,1] - y[0,2] - y[0,3] - y[0,4] - y[0,5] - y[0,6] - y[0,7] - y[0,8] - y[0,9]
        elif t==1:
            error += -1* y[0,0] + 1 - y[0,1] - y[0,2] - y[0,3] - y[0,4] - y[0,5] - y[0,6] - y[0,7] - y[0,8] - y[0,9]
        elif t==2:
            error += -1* y[0,0] - y[0,1] + 1 - y[0,2] - y[0,3] - y[0,4] - y[0,5] - y[0,6] - y[0,7] - y[0,8] - y[0,9]
        elif t==3:
            error += -1* y[0,0] - y[0,1] - y[0,2] + 1 - y[0,3] - y[0,4] - y[0,5] - y[0,6] - y[0,7] - y[0,8] - y[0,9]
        elif t==4:
            error += -1 * y[0, 0] - y[0, 1] - y[0, 2] - y[0, 3] +1- y[0, 4] - y[0, 5] - y[0, 6] - y[0, 7] - y[0, 8] - y[0, 9]
        elif t==5:
            print('DONE ***')
            error += -1 * y[0, 0] - y[0, 1] - y[0, 2] - y[0, 3] - y[0, 4] +1- y[0, 5] - y[0, 6] - y[0, 7] - y[0, 8] - y[0, 9]
        elif t==6:
            error += -1 * y[0, 0] - y[0, 1] - y[0, 2] - y[0, 3] - y[0, 4] - y[0, 5] +1-y[0, 6] - y[0, 7] - y[0, 8] -y[0, 9]
        elif t==7:
            error += -1 * y[0, 0] - y[0, 1] - y[0, 2] - y[0, 3] - y[0, 4] - y[0, 5] - y[0, 6] +1- y[0, 7] - y[0, 8] - y[0, 9]
        elif t==8:
            error += -1 * y[0, 0] - y[0, 1] - y[0, 2] - y[0, 3] - y[0, 4] - y[0, 5] - y[0, 6] - y[0, 7] +1- y[0, 8] - y[0, 9]
        elif t==9:
            error += -1 * y[0, 0] - y[0, 1] - y[0, 2] - y[0, 3] - y[0, 4] - y[0, 5] - y[0, 6] - y[0, 7] - y[0, 8] +1- y[0, 9]
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


    def training(self):
      for epoch in range(self.N_Epochs):
        for i in range(self.train.shape[0]):
          y,total_act_val_layers=self.feedforward(self.train[i, :])
          #print(y)
          total_gradient2,total_gradient,gradient_outputlayer=self.Backpropagate(y,self.target_train[i],total_act_val_layers)
          print(total_gradient2)
          total_weights_hidden,weights_of_output=self.update_weights(total_gradient,gradient_outputlayer)
          print('\nhidden weights updated = \n',total_weights_hidden,'\n')
          print('output layer weights updated = \n', weights_of_output)
          print('*****************************************************************************************\n')





    # def testing(self):
    #
    #     C0_C0=0
    #     C0_C1=0
    #     C0_C2=0
    #     C0_C3=0
    #     C0_C4=0
    #     C0_C5=0
    #     C0_C6=0
    #     C0_C7=0
    #     C0_C8=0
    #     C0_C9=0
    #
    #     C1_C0=0
    #     C1_C1=0
    #     C1_C2=0
    #     C1_C3=0
    #     C1_C4=0
    #     C1_C5=0
    #     C1_C6=0
    #     C1_C7=0
    #     C1_C8=0
    #     C1_C9=0
    #
    #     C2_C0=0
    #     C2_C1=0
    #     C2_C2=0
    #     C2_C3=0
    #     C2_C4=0
    #     C2_C5=0
    #     C2_C6=0
    #     C2_C7=0
    #     C2_C8=0
    #     C2_C9=0
    #
    #     C3_C0=0
    #     C3_C1=0
    #     C3_C2=0
    #     C3_C3=0
    #     C3_C4=0
    #     C3_C5=0
    #     C3_C6=0
    #     C3_C7=0
    #     C3_C8=0
    #     C3_C9=0
    #
    #     C4_C0=0
    #     C4_C1=0
    #     C4_C2=0
    #     C4_C3=0
    #     C4_C4=0
    #     C4_C5=0
    #     C4_C6=0
    #     C4_C7=0
    #     C4_C8=0
    #     C4_C9=0
    #
    #     C5_C0=0
    #     C5_C1=0
    #     C5_C2=0
    #     C5_C3=0
    #     C5_C4=0
    #     C5_C5=0
    #     C5_C6=0
    #     C5_C7=0
    #     C5_C8=0
    #     C5_C9=0
    #
    #     C6_C0=0
    #     C6_C1=0
    #     C6_C2=0
    #     C6_C3=0
    #     C6_C4=0
    #     C6_C5=0
    #     C6_C6=0
    #     C6_C7=0
    #     C6_C8=0
    #     C6_C9=0
    #
    #     C7_C0=0
    #     C7_C1=0
    #     C7_C2=0
    #     C7_C3=0
    #     C7_C4=0
    #     C7_C5=0
    #     C7_C6=0
    #     C7_C7=0
    #     C7_C8=0
    #     C7_C9=0
    #
    #     C8_C0=0
    #     C8_C1=0
    #     C8_C2=0
    #     C8_C3=0
    #     C8_C4=0
    #     C8_C5=0
    #     C8_C6=0
    #     C8_C7=0
    #     C8_C8=0
    #     C8_C9=0
    #
    #     C9_C0=0
    #     C9_C1=0
    #     C9_C2=0
    #     C9_C3=0
    #     C9_C4=0
    #     C9_C5=0
    #     C9_C6=0
    #     C9_C7=0
    #     C9_C8=0
    #     C9_C9=0
    #
    #
    #     for i in range (self.test.shape[0]):
    #         y,total_act=self.feedforward(self.test[i,:])
    #         if self.target_test[i] == 0:
    #
    #             if y[0,0]>y[0,1] and y[0,0]>y[0,2]and y[0,0]>y[0,3] and y[0,0]>y[0,4] and y[0,0]>y[0,5] and y[0,0]>y[0,6]\
    #                     and y[0,0]>y[0,7]and y[0,0]>y[0,8]and y[0,0]>y[0,9]:
    #                 C0_C0+=1
    #             elif y[0,1] > y[0,0] and y[0,1] > y[0,2]and y[0,1] > y[0,3] and y[0,1] > y[0,4]and y[0,1] > y[0,5]and y[0,1] > y[0,6]and y[0,1] > y[0,7]and y[0,1] > y[0,8]and y[0,1] > y[0,9]:
    #                 C0_C1+=1
    #             elif y[0,2]> y[0,0]  and y[0,2]> y[0,1]and y[0,2]> y[0,3]and y[0,2]> y[0,4]and y[0,2]> y[0,5]and y[0,2]> y[0,6]and y[0,2]> y[0,7]and y[0,2]> y[0,8]and y[0,2]> y[0,9] :
    #                 C0_C2+=1
    #             elif y[0,3]>y[0,0] and y[0,3]>y[0,1] and y[0,3]>y[0,2] and y[0,3]>y[0,4] and y[0,3]>y[0,5]and y[0,3]>y[0,6]and y[0,3]>y[0,7]and y[0,3]>y[0,8]and y[0,3]>y[0,9]:
    #                 C0_C3+=1
    #             elif y[0,4]>y[0,0] and y[0,4]>y[0,1] and y[0,4]>y[0,2] and y[0,4]>y[0,3]and y[0,4]>y[0,5]and y[0,4]>y[0,6]and y[0,4]>y[0,7]and y[0,4]>y[0,8]and y[0,4]>y[0,9]:
    #                 C0_C4+=1
    #             elif y[0,5]>y[0,0] and y[0,5]>y[0,1] and y[0,5]>y[0,2]and y[0,5]>y[0,3]and y[0,5]>y[0,4]and y[0,5]>y[0,6]and y[0,5]>y[0,7]and y[0,5]>y[0,8]and y[0,5]>y[0,9]:
    #                 C0_C5+=1
    #             elif y[0,6]>y[0,0] and y[0,6]>y[0,1] and y[0,6]>y[0,2] and y[0,6]>y[0,3]and y[0,6]>y[0,4]and y[0,6]>y[0,5]and y[0,6]>y[0,7]and y[0,6]>y[0,8]and y[0,6]>y[0,9]:
    #                 C0_C6+=1
    #             elif y[0,7]>y[0,0] and y[0,7]>y[0,1] and y[0,7]>y[0,2]and y[0,7]>y[0,3]and y[0,7]>y[0,4]and y[0,7]>y[0,5]and y[0,7]>y[0,6]and y[0,7]>y[0,8]and y[0,7]>y[0,9]:
    #                 C0_C7+=1
    #             elif y[0,8]>y[0,0] and y[0,8]>y[0,1] and y[0,8]>y[0,2]and y[0,8]>y[0,3]and y[0,8]>y[0,4]and y[0,8]>y[0,5]and y[0,8]>y[0,6]and y[0,8]>y[0,7]and y[0,8]>y[0,9]:
    #                 C0_C8+=1
    #             elif y[0,9]>y[0,0] and y[0,9]>y[0,1]and y[0,9]>y[0,2]and y[0,9]>y[0,3]and y[0,9]>y[0,4]and y[0,9]>y[0,5]and y[0,9]>y[0,6]and y[0,9]>y[0,7]and y[0,9]>y[0,8]:
    #                 C0_C9+=1
    #
    #         elif self.target_test[i] == 1:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C1_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C1_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C1_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C1_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C1_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C1_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C1_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C1_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C1_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C1_C9 += 1
    #
    #
    #         elif self.target_test[i] == 2:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C2_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C2_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C2_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C2_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C2_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C2_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C2_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C2_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C2_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C2_C9 += 1
    #
    #         elif self.target_test[i] == 3:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[0, 5]\
    #                     and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C3_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C3_C1+= 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C3_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C3_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C3_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C3_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C3_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C3_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C3_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C3_C9 += 1
    #
    #
    #         elif self.target_test[i] == 4:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] \
    #                     and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C4_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C4_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C4_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C4_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C4_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C4_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C4_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C4_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C4_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C4_C9 += 1
    #
    #         elif self.target_test[i] == 5:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] \
    #                     and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C4_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C5_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C5_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C5_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C5_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C5_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C5_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C5_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C5_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C5_C9 += 1
    #
    #         elif self.target_test[i] == 6:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] \
    #                     and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C6_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C6_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C6_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C6_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C6_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C6_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C6_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C6_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C6_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C6_C9 += 1
    #
    #         elif self.target_test[i] == 7:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] \
    #                     and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C7_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C7_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C7_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C7_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C7_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C7_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C7_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C7_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C7_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C7_C9 += 1
    #
    #         elif self.target_test[i] == 8:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] \
    #                     and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C8_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C8_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C8_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C8_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C8_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C8_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C8_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C8_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C8_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C8_C9 += 1
    #
    #         elif self.target_test[i] == 9:
    #             if y[0, 0] > y[0, 1] and y[0, 0] > y[0, 2] and y[0, 0] > y[0, 3] and y[0, 0] > y[0, 4] and y[0, 0] > y[
    #                 0, 5] \
    #                     and y[0, 0] > y[0, 6] and y[0, 0] > y[0, 7] and y[0, 0] > y[0, 8] and y[0, 0] > y[0, 9]:
    #                 C9_C0 += 1
    #             elif y[0, 1] > y[0, 0] and y[0, 1] > y[0, 2] and y[0, 1] > y[0, 3] and y[0, 1] > y[0, 4] and y[0, 1] > \
    #                     y[0, 5] and y[0, 1] > y[0, 6] and y[0, 1] > y[0, 7] and y[0, 1] > y[0, 8] and y[0, 1] > y[0, 9]:
    #                 C9_C1 += 1
    #             elif y[0, 2] > y[0, 0] and y[0, 2] > y[0, 1] and y[0, 2] > y[0, 3] and y[0, 2] > y[0, 4] and y[0, 2] > \
    #                     y[0, 5] and y[0, 2] > y[0, 6] and y[0, 2] > y[0, 7] and y[0, 2] > y[0, 8] and y[0, 2] > y[0, 9]:
    #                 C9_C2 += 1
    #             elif y[0, 3] > y[0, 0] and y[0, 3] > y[0, 1] and y[0, 3] > y[0, 2] and y[0, 3] > y[0, 4] and y[0, 3] > \
    #                     y[0, 5] and y[0, 3] > y[0, 6] and y[0, 3] > y[0, 7] and y[0, 3] > y[0, 8] and y[0, 3] > y[0, 9]:
    #                 C9_C3 += 1
    #             elif y[0, 4] > y[0, 0] and y[0, 4] > y[0, 1] and y[0, 4] > y[0, 2] and y[0, 4] > y[0, 3] and y[0, 4] > \
    #                     y[0, 5] and y[0, 4] > y[0, 6] and y[0, 4] > y[0, 7] and y[0, 4] > y[0, 8] and y[0, 4] > y[0, 9]:
    #                 C9_C4 += 1
    #             elif y[0, 5] > y[0, 0] and y[0, 5] > y[0, 1] and y[0, 5] > y[0, 2] and y[0, 5] > y[0, 3] and y[0, 5] > \
    #                     y[0, 4] and y[0, 5] > y[0, 6] and y[0, 5] > y[0, 7] and y[0, 5] > y[0, 8] and y[0, 5] > y[0, 9]:
    #                 C9_C5 += 1
    #             elif y[0, 6] > y[0, 0] and y[0, 6] > y[0, 1] and y[0, 6] > y[0, 2] and y[0, 6] > y[0, 3] and y[0, 6] > \
    #                     y[0, 4] and y[0, 6] > y[0, 5] and y[0, 6] > y[0, 7] and y[0, 6] > y[0, 8] and y[0, 6] > y[0, 9]:
    #                 C9_C6 += 1
    #             elif y[0, 7] > y[0, 0] and y[0, 7] > y[0, 1] and y[0, 7] > y[0, 2] and y[0, 7] > y[0, 3] and y[0, 7] > \
    #                     y[0, 4] and y[0, 7] > y[0, 5] and y[0, 7] > y[0, 6] and y[0, 7] > y[0, 8] and y[0, 7] > y[0, 9]:
    #                 C9_C7 += 1
    #             elif y[0, 8] > y[0, 0] and y[0, 8] > y[0, 1] and y[0, 8] > y[0, 2] and y[0, 8] > y[0, 3] and y[0, 8] > \
    #                     y[0, 4] and y[0, 8] > y[0, 5] and y[0, 8] > y[0, 6] and y[0, 8] > y[0, 7] and y[0, 8] > y[0, 9]:
    #                 C9_C8 += 1
    #             elif y[0, 9] > y[0, 0] and y[0, 9] > y[0, 1] and y[0, 9] > y[0, 2] and y[0, 9] > y[0, 3] and y[0, 9] > \
    #                     y[0, 4] and y[0, 9] > y[0, 5] and y[0, 9] > y[0, 6] and y[0, 9] > y[0, 7] and y[0, 9] > y[0, 8]:
    #                 C9_C9 += 1
    #
    #     return np.array([[C0_C0, C0_C1, C0_C2, C0_C3, C0_C4, C0_C5, C0_C6, C0_C7, C0_C8, C0_C9], [C1_C0, C1_C1, C1_C2, C1_C3, C1_C4, C1_C5, C1_C6, C1_C7, C1_C8, C1_C9]
    #             ,[C2_C0, C2_C1, C2_C2, C2_C3, C2_C4, C2_C5, C2_C6, C2_C7, C2_C8, C2_C9],[C3_C0, C3_C1, C3_C2, C3_C3, C3_C4, C3_C5, C3_C6, C3_C7, C3_C8, C3_C9 ]
    #             ,[C4_C0, C4_C1, C4_C2, C4_C3, C4_C4, C4_C5, C4_C6, C4_C7, C4_C8, C4_C9],[C5_C0, C5_C1, C5_C2, C5_C3, C5_C4, C5_C5, C5_C6, C5_C7, C5_C8, C5_C9]
    #             ,[C6_C0, C6_C1, C6_C2, C6_C3, C6_C4, C6_C5, C6_C6, C6_C7, C6_C8, C6_C9],[C7_C0, C7_C1, C7_C2, C7_C3, C7_C4, C7_C5, C7_C6, C7_C7, C7_C8, C7_C9]
    #             ,[C8_C0, C8_C1, C8_C2, C8_C3, C8_C4, C8_C5, C8_C6, C8_C7, C8_C8, C8_C9],[C9_C0, C9_C1, C9_C2, C9_C3, C9_C4, C9_C5, C9_C6, C9_C7, C9_C8, C9_C9]])
    #
    #
    #
    #

    def testing(self):

        conf_matrix = [[0 for i in range(10)], [0 for i in range(10)], [0 for i in range(10)]\
            ,[0 for i in range(10)],[0 for i in range(10)],[0 for i in range(10)]\
            ,[0 for i in range(10)],[0 for i in range(10)],[0 for i in range(10)],[0 for i in range(10)]]

        # print(conf_matrix)
        # print(conf_matrix[0][0])
        # print(test.shape[0])
        for i in range(self.test.shape[0]):
            y, total_act = self.feedforward(self.test[i, :])
            max_neuron = max(y[0, 0], y[0, 1], y[0, 2],y[0,3],y[0,4] ,y[0,5], y[0,6],y[0,7],y[0,8],y[0,9])
            # print('max = ',max_neuron)
            # print('target is = ',t_test[i,0])
            if self.target_test[i] == 0:

                if max_neuron == y[0, 0]:
                    conf_matrix[0][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[0][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[0][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[0][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[0][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[0][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[0][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[0][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[0][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[0][9] += 1

            elif self.target_test[i] == 1:
                if max_neuron == y[0, 0]:
                    conf_matrix[1][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[1][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[1][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[1][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[1][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[1][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[1][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[1][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[1][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[1][9] += 1




            elif self.target_test[i] == 2:
                if max_neuron == y[0, 0]:
                    conf_matrix[2][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[2][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[2][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[2][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[2][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[2][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[2][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[2][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[2][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[2][9] += 1



            elif self.target_test[i] == 3:
                if max_neuron == y[0, 0]:
                    conf_matrix[3][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[3][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[3][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[3][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[3][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[3][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[3][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[3][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[3][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[3][9] += 1

            elif self.target_test[i] == 4:
                if max_neuron == y[0, 0]:
                    conf_matrix[4][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[4][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[4][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[4][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[4][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[4][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[4][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[4][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[4][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[4][9] += 1

            elif self.target_test[i] == 5:
                if max_neuron == y[0, 0]:
                    conf_matrix[5][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[5][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[5][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[5][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[5][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[5][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[5][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[5][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[5][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[5][9] += 1

            elif self.target_test[i] == 6:
                if max_neuron == y[0, 0]:
                    conf_matrix[6][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[6][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[6][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[6][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[6][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[6][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[6][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[6][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[6][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[6][9] += 1



            elif self.target_test[i] == 7:
                if max_neuron == y[0, 0]:
                    conf_matrix[7][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[7][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[7][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[7][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[7][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[7][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[7][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[7][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[7][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[7][9] += 1




            elif self.target_test[i] == 8:
                if max_neuron == y[0, 0]:
                    conf_matrix[8][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[8][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[8][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[8][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[8][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[8][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[8][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[8][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[8][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[8][9] += 1

            elif self.target_test[i] == 9:
                if max_neuron == y[0, 0]:
                    conf_matrix[9][0] += 1
                elif max_neuron == y[0, 1]:
                    conf_matrix[9][1] += 1
                elif max_neuron == y[0, 2]:
                    conf_matrix[9][2] += 1
                elif max_neuron == y[0, 3]:
                    conf_matrix[9][3] += 1
                elif max_neuron == y[0, 4]:
                    conf_matrix[9][4] += 1
                elif max_neuron == y[0, 5]:
                    conf_matrix[9][5] += 1
                elif max_neuron == y[0, 6]:
                    conf_matrix[9][6] += 1
                elif max_neuron == y[0, 7]:
                    conf_matrix[9][7] += 1
                elif max_neuron == y[0, 8]:
                    conf_matrix[9][8] += 1
                elif max_neuron == y[0, 9]:
                    conf_matrix[9][9] += 1


        # print("conf list = \n",conf_matrix)
        return np.array(conf_matrix)


    def classify(self):


        self.training()
        print('original hidden weighs = \n',self.total_weights_hidden)
        print('original output weighs = \n', self.weights_of_output)
        Confusion_Matrix=self.testing()
        print('conf matrix is = \n',Confusion_Matrix)
        print("accuracy is = ", ((Confusion_Matrix[0][0] + Confusion_Matrix[1][1]+ Confusion_Matrix[2][2]+ Confusion_Matrix[3][3]+ Confusion_Matrix[4][4]+ Confusion_Matrix[5][5]+ Confusion_Matrix[6][6]+ Confusion_Matrix[7][7]+ Confusion_Matrix[8][8]+Confusion_Matrix[9][9]) / len(self.test)) * 100, "%")
