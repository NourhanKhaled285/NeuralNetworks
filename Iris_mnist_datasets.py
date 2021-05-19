import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from Perceptron import *
from Adaline import *
from BackProbagation import *
# from BackProbagation_mnist_dataset import *

from PIL import ImageTk,Image


#loading data

def read_data():
 with open('IrisData.txt') as f:
     lines = f.readlines()
 f.close()
 dataset_X = np.empty([150,4])
 for i in range (len(lines)):
   if i!=0:
     dataset_X[i-1] = lines[i].split(',')[:4]
 return dataset_X



# def read_data2():
#     data_train = pd.read_csv('mnist_train.csv')
#     train = data_train.iloc[:, 1:data_train.shape[1] + 1]
#     target_train = data_train['label']
#     train = np.array(train)
#
#
#     data_test = pd.read_csv('mnist_test.csv')
#     test = data_test.iloc[:, 1:data_test.shape[1] + 1]
#     target_test = data_test['label']
#     test = np.array(test)
#     return train,test,target_train,target_test

# print('train data \n',read_data())



#Plot_data
def plot(X_axis,Y_axis,dataset_X,y1,y2):
 plt.figure(X_axis+'_'+Y_axis)
 plt.scatter(dataset_X[:50,y1],dataset_X[:50,y2])
 plt.scatter(dataset_X[50:100,y1],dataset_X[50:100,y2])
 plt.scatter(dataset_X[100:150, y1], dataset_X[100:150, y2])

 plt.xlabel(X_axis)
 plt.ylabel(Y_axis)
 plt.show()


Dataset_Of_Features = read_data()
# train,test,target_train,target_test=read_data2()
plot("X1", "X2", Dataset_Of_Features, 0, 1)
plot("X1", "X3", Dataset_Of_Features, 0, 2)
plot("X1", "X4", Dataset_Of_Features, 0, 3)
plot("X2", "X3", Dataset_Of_Features, 1, 2)
plot("X2", "X4", Dataset_Of_Features, 1, 3)
plot("X3", "X4", Dataset_Of_Features, 2, 3)


# print(Dataset_Of_Features)
# print(Dataset_Of_Features[:3,:3:2])


# conv=Canvas(top,width=500,height=1200)
# img=ImageTk.PhotoImage(Image.open('bc.jpg'))
# conv.create_image(0,0,anchor=NW,image=img)
# conv.pack()





def run_backprobagation():
    back_ProbForm = Tk()
    back_ProbForm.geometry('700x500')
    back_ProbForm.config(bg='#E0B0FF')
    def show1(event):
        back_ProbForm.destroy()
        MainForm.deiconify()

    variable_Lhidden_layers = StringVar(back_ProbForm)
    variable_Lneurons = StringVar(back_ProbForm)
    variable_Lepochs = StringVar(back_ProbForm)
    variable_Lrate = StringVar(back_ProbForm)
    variable_Lback=StringVar(back_ProbForm)
    activation_type=StringVar(back_ProbForm)
    var_txt = StringVar(back_ProbForm)
    Cbias = IntVar()

    variable_Lhidden_layers.set("Enter number of hidden layers")
    variable_Lneurons.set("Enter number of neurons for each layer")
    variable_Lepochs.set("Enter number of epochs")
    variable_Lrate.set("Enter learning rate")
    activation_type.set("Select Activation Type")
    variable_Lback.set("Back")
    var_txt.set('seperated by spaces')


    Lepochs = Label(back_ProbForm, textvariable=variable_Lepochs, bg='#E0B0FF', fg='#FFFF00', font='Andalus 12 italic bold')
    Lrate = Label(back_ProbForm, textvariable=variable_Lrate, bg='#E0B0FF', fg='#FFFF00', font='Andalus 12 italic bold')
    Lhiddenlayers=Label(back_ProbForm, textvariable=variable_Lhidden_layers, bg='#E0B0FF', fg='#FFFF00', font='Andalus 12 italic bold')
    Lneurons=Label(back_ProbForm, textvariable=variable_Lneurons, bg='#E0B0FF', fg='#FFFF00', font='Andalus 12 italic bold')
    Lback = Label(back_ProbForm, textvariable=variable_Lback, bg='#E0B0FF', fg='#FFFF00', font='Andalus 15 italic bold')
    Lback.pack()
    Lback.bind("<Button>", show1)

    hidden_layers_txt =Entry(back_ProbForm,width=29,borderwidth=2)
    neurons_txt = Entry(back_ProbForm,width=25,borderwidth=2,textvariable=var_txt ,fg='#595959', font='Andalus 10 italic')
    lear_rate_txt = Entry(back_ProbForm,width=29,borderwidth=2)
    epochs_txt = Entry(back_ProbForm,width=29,borderwidth=2)

    Checkbias = Checkbutton(back_ProbForm, bg='#E0B0FF', fg='#FFFF00', text="Bias", font='bold', variable=Cbias)
    activationType_menu = OptionMenu(back_ProbForm,activation_type,"Sigmoid","Hyperbolic Tangent sigmoid")
    activationType_menu.config(bg='#FFFF00')


    Lhiddenlayers.place(x=20,y=90)
    Lneurons.place(x=20,y=130)
    Lepochs.place(x=20, y=170)
    Lrate.place(x=20, y=210)
    Lback.place(x=640, y=430)

    hidden_layers_txt.place(x=348,y=90)
    neurons_txt.place(x=348,y=130)
    lear_rate_txt.place(x=348, y=210)
    epochs_txt.place(x=348, y=170)

    Checkbias.place(x=20, y=290)

    activationType_menu.place(x=20,y=240)

    def run_backprobagationModel_mnist():
        n_neurons=list((neurons_txt.get()).split(' '))

        back = BackProbagation_mnist(train,test,target_train,target_test,int(hidden_layers_txt.get()),n_neurons,float(lear_rate_txt.get()),int(epochs_txt.get()),activation_type.get(),Cbias.get())
        back.classify()


    def run_backprobagationModel_iris():
        n_neurons = list((neurons_txt.get()).split(' '))
        back = BackProbagation(Dataset_Of_Features,int(hidden_layers_txt.get()), n_neurons, float(lear_rate_txt.get()),int(epochs_txt.get()), activation_type.get(), Cbias.get())
        back.classify()

    backprobagation_mnsit_btn = Button(back_ProbForm, bg='#FFFF00', text="Backprobagation mnist DataSet", width=25,
                                  command=run_backprobagationModel_mnist)
    backprobagation_iris_btn = Button(back_ProbForm, bg='#FFFF00', text="Backprobagation Iris DataSet", width=25,
                                       command=run_backprobagationModel_iris)
    backprobagation_mnsit_btn.place(x=350, y=300)
    backprobagation_iris_btn.place(x=150, y=300)
    MainForm.withdraw()
    back_ProbForm.mainloop()






def linear_classification():
    MainForm.withdraw()
    top = Tk()
    top.geometry('700x500')
    top.config(bg='#E0B0FF')


    def show1(event):
        top.destroy()
        MainForm.deiconify()

    Cbias = IntVar()
    variable_Feature = StringVar(top)
    variable_classes = StringVar(top)
    variable_Lepochs = StringVar(top)
    variable_Lrate = StringVar(top)
    variable_threshold= StringVar(top)
    variable_Lback=StringVar(top)

    variable_Feature.set("X1_X2")
    variable_classes.set("C1&C2")
    variable_Lepochs.set("Enter number of epochs")
    variable_Lrate.set("Enter learning rate")
    variable_threshold.set("Enter threshold")
    variable_Lback.set("Back")

    MI_Feature = OptionMenu(top,variable_Feature,"X1_X2","X1_X3","X1_X4","X2_X3","X2_X4","X3_X4")
    MI_Feature.config(bg='#FFFF00')
    MI_classes = OptionMenu(top,variable_classes,"C1&C2","C1&C3","C2&C3")
    MI_classes.config(bg='#FFFF00')

    Lepochs = Label(top,textvariable=variable_Lepochs,bg='#E0B0FF', fg='#FFFF00',font='Andalus 12 italic bold')
    Lrate = Label(top,textvariable=variable_Lrate,bg='#E0B0FF', fg='#FFFF00',font='Andalus 12 italic bold')
    Lthreshold=Label(top,textvariable=variable_threshold,bg='#E0B0FF', fg='#FFFF00',font='Andalus 12 italic bold')
    Lback = Label(top, textvariable=variable_Lback, bg='#E0B0FF', fg='#FFFF00', font='Andalus 15 italic bold')
    Lback.pack()
    Lback.bind("<Button>",show1)


    Eepochs = Entry(top,width=30)
    Erate = Entry(top,width=30)
    Threshold_txt=Entry(top,width=30)
    Checkbias = Checkbutton(top,bg='#E0B0FF', fg='#FFFF00', text="Bias",font='bold',variable=Cbias)

    def collect_data():
        Feature = variable_Feature.get()
        Classes = variable_classes.get()
        Number_Of_Epochs = Eepochs.get()
        Learing_rate = Erate.get()
        Bias = Cbias.get()
        threshold = Threshold_txt.get()
        return Feature, Classes, Number_Of_Epochs, Learing_rate, Bias, threshold

    def run_Perceptron():
        Feature, Classes, Number_Of_Epochs, Learing_rate, Bias, threshold = collect_data()
        p = Perceptron(Dataset_Of_Features, int(Number_Of_Epochs), float(Learing_rate), Classes, Feature, Bias)
        p.classify()


    def run_Adaline():
        Feature, Classes, Number_Of_Epochs, Learing_rate, Bias, threshold = collect_data()
        adaline = Adaline(Dataset_Of_Features, int(Number_Of_Epochs), float(Learing_rate), Classes, Feature,
                          float(threshold), Bias)
        adaline.classify()


    Perceptron_btn = Button(top,bg='#FFFF00', text="classifiy using perceptron", width=20, command = run_Perceptron)
    Adaline_btn = Button(top, bg='#FFFF00',text="classifiy using Adaline", width=20, command = run_Adaline)


    MI_Feature.place(x=20,y=10)
    MI_classes.place(x=20,y=50)
    Lepochs.place(x=20,y=90)
    Eepochs.place(x=220,y=90)
    Lrate.place(x=20,y=130)
    Lthreshold.place(x=20,y=170)
    Lback.place(x=640,y=430)

    Erate.place(x=220,y=130)
    Threshold_txt.place(x=220,y=170)
    Checkbias.place(x=20,y=220)
    Perceptron_btn.place(x=100, y=250)
    Adaline_btn.place (x=300, y=250)




MainForm = Tk()
MainForm.geometry('500x500')
MainForm.config(bg='#E0B0FF')
linear_class_btn = Button(MainForm,bg='#FFFF00', text="Using Linear Classification", width=20, command = linear_classification)
nonlinear_class_btn=Button(MainForm,bg='#FFFF00', text="Using Non Linear Classification", width=25, command = run_backprobagation)
linear_class_btn.place(x=100, y=200)
nonlinear_class_btn.place(x=300,y=200)
MainForm.mainloop()


#Function_collectdata that is the action pf button, the start of program,#####line_41








