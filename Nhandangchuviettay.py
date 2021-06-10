from tkinter import Tk, Frame, Scrollbar, Label, END, Entry, Text, VERTICAL, Button,Canvas,PhotoImage
import socket
from tkinter import filedialog
from PIL import ImageTk,Image  
from tkinter import messagebox
import numpy
import matplotlib
import glob
import imageio 
import matplotlib.pyplot as plt
import pylab

class GUI:
        def __init__(self, master):
            self.png=''
            self.input_nodes = 784
            self.hidden_nodes = 100
            self.output_nodes = 10
            self.learning_rate = 0.1
            
            self.iinput_nodes = 784
            self.ihidden_nodes1 = 200
            self.ihidden_nodes2 = 200
            self.ioutput_nodes = 26
            self.ilearning_rate = 0.01
            
            self.root = master
            self.frameResult = Frame(background="blue")
            self.label=None
            self.check=True
            self.join_button = None
            self.initialize_gui()
            
        def initialize_gui(self):
            self.root.title("Nhận dạng chữ cái viết tay")
            self.root.geometry("500x250+300+300")
            self.root.configure(background='white')
            self.root.resizable(0, 0)
            self.display_name_section()
            
        def display_name_section(self):
            frame = Frame()
            self.join_button = Button(frame, text="Chọn ảnh", width=10, command=self.onOpen,bg='silver').pack(side='left')
            frame.pack(side='top', anchor='nw')
            frame = Frame()
            
            self.join_button = Button(frame, text="Reset", width=10, command=self.onDelete,padx=20,pady=5,font=("Serif", 12),bg='silver').pack(side='bottom')
            Label(frame, text='Reset                        ', font=("Serif", 12),bg='white').pack(side='bottom')
            
            self.join_button = Button(frame, text="Start", width=10, command=self.onTrain2,padx=20,pady=5,font=("Serif", 12),bg='silver').pack(side='bottom')
            Label(frame, text='Nhận dạng chữ cái   ', font=("Serif", 12),bg='white').pack(side='bottom')
            self.join_button = Button(frame, text="Start", width=10, command=self.onTrain,padx=20,pady=5,font=("Serif", 12),bg='silver').pack(side='bottom')
            Label(frame, text='Nhận dạng chữ số    ', font=("Serif", 12),bg='white').pack(side='bottom')
            
            frame.pack(side='left')
            
        def onDelete(self):
            self.frameResult.destroy()
            self.label.destroy()
            
        def onOpen(self):
            self.frameResult.destroy()
            if(self.label!=None):    
                self.label.destroy()
            self.frameResult = Frame()
            filename =  filedialog.askopenfilename(title = "png",filetypes = (("PNG files","*.png"),("all files","*.*")))
            self.png=filename
            print(self.png)
            
        def onTrain(self):            
            from network_class import neuralNetwork
            n = neuralNetwork(self.input_nodes,self.hidden_nodes,self.output_nodes,self.learning_rate)
            n.load()

            test_data_file = open("test_10.csv", 'r') 
            test_data_list = test_data_file.readlines() 
            test_data_file.close() 


            scorecard = [] 

            for record in test_data_list: 
                all_values = record.split(',') 
                correct_label = int( all_values[ 0]) 
                inputs = (numpy.asfarray( all_values[ 1:]) / 255.0 * 0.99) + 0.01 
                outputs = n.query( inputs) 
                label = numpy.argmax( outputs) 
                if (label == correct_label): 
                    scorecard.append( 1) 
                else: 
                    scorecard.append( 0) 
                    pass 
                pass 

            our_own_dataset = []

            for image_file_name in glob.glob(self.png):
                
                label = int(image_file_name[-5:-4])
                
                print ("loading ... ", image_file_name)
                img_array = imageio.imread(image_file_name, as_gray=True)
                
                img_data  = 255.0 - img_array.reshape(784)
                
                img_data = (img_data / 255.0 * 0.99) + 0.01
                print(numpy.min(img_data))
                print(numpy.max(img_data))
                
                record = numpy.append(label,img_data)
                our_own_dataset.append(record)
                
                pass
            item = 0

            plt.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

            correct_label = our_own_dataset[item][0]

            inputs = our_own_dataset[item][1:]

            outputs = n.query(inputs)
            print (outputs)

            label = numpy.argmax(outputs)
            print("Dự đoán ", label)
            if (label == correct_label):
                print ("match!")
            else:
                print ("no match!")
                pass
                  
            t1="Số: "+str(label)+'              '
            t2='​Độ chính xác: '+str(int(outputs[label]*100))+'% '
            Label(self.frameResult, text=t1, font=("Serif", 14)).pack(side='top', anchor='w')
            Label(self.frameResult, text=t2, font=("Serif", 14)).pack(side='top', anchor='w')
            self.frameResult.pack(side='top')
            image = Image.open(self.png)
            image = image.resize((100, 100), Image.ANTIALIAS) 
            photo = ImageTk.PhotoImage(image)
            self.label = Label(image=photo)
            self.label.image = photo
            self.label.pack()
            
        def onTrain2(self):
            from network import neuralNetwork
            n = neuralNetwork(self.iinput_nodes,self.ihidden_nodes1, self.ihidden_nodes2,self.ioutput_nodes, self.ilearning_rate)
            n.load()

            our_own_dataset = []

            for image_file_name in glob.glob(self.png):
                
                label = int(image_file_name[-5:-4])
                
                print ("loading ... ", image_file_name)
                img_array = imageio.imread(image_file_name, as_gray=True)
                
                img_data  = 255.0 - img_array.reshape(784)
                
                img_data = (img_data / 255.0 * 0.99) + 0.01
                print(numpy.min(img_data))
                print(numpy.max(img_data))
                
                record = numpy.append(label,img_data)
                our_own_dataset.append(record)
                pass

            arr = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O","P","Q","R","S","T","U","V","W","X","Y","Z"]
            item = 0
            matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')
            correct_label = our_own_dataset[item][0]
            inputs = our_own_dataset[item][1:]
            outputs = n.query(inputs) 
            print (outputs)

            kitu=''
            label = numpy.argmax(outputs)
            for i, j in enumerate(arr):
                if(label == i):
                    print("Dự đoán: ", j)
                    kitu=j
                    break;
                
            t1="Chữ cái:  "+j+'             '
            t2='​Độ chính xác: '+str(int(outputs[label]*100))+'% '
            Label(self.frameResult, text=t1, font=("Serif", 14)).pack(side='top', anchor='w')
            Label(self.frameResult, text=t2, font=("Serif", 14)).pack(side='top', anchor='w')
            self.frameResult.pack(side='top')
            image = Image.open(self.png)
            image = image.resize((100, 100), Image.ANTIALIAS) 
            photo = ImageTk.PhotoImage(image)
            self.label = Label(image=photo)
            self.label.image = photo 
            self.label.pack()

if __name__ == '__main__':
    root = Tk()
    gui = GUI(root)
    root.mainloop()
    
