# This script is a simple drawing application that uses the Tkinter library to create a GUI that allows users to draw on a canvas. The script also has a "Predict" button that, when clicked, will take the current drawing and use a pre-trained deep learning model to predict the digit that the user drew. The script also uses the PIL, numpy, cv2, matplotlib, sklearn and keras library to process the image and make the prediction. The script loads the pre-trained model from a file and uses kmeans to segment the image before making the prediction. The predicted digit is then displayed on the GUI using a label.

# Tkinter for app and matplotlib to show
from tkinter import Tk, Canvas, Button, Label, font
import matplotlib.pyplot as plt

# Loading and preprocessing image
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

# To load pre-trained mnist model
from keras.models import load_model

# Our global variables to change easilly 
WIDTH = 480
HEIGHT = 480
BACKGROUND_COLOR = 'white'
DRAW_COLOR = 'black'
FONT = ("Arial", 14, "bold")
FONT_COLOR = 'gray8'
BUT1_COLOR = 'medium orchid'
BUT2_COLOR = 'cornflower blue'
IMAGE_NAME = 'digit'
MODEL = '/home/adam/Development/\
Machine_learning/Study/Hands-on-\
Machine-Learning/11. training_\
deep_neural_networks/mnist.h5'

# Main Class
class DrawingApplication:
    def __init__(self):
        # App with title
        self.root = Tk()
        self.root.title("Drawing Application")

        # Canvas with size and bg color in which you can draw using left mouse button
        self.canvas = Canvas(self.root, width=WIDTH, height=HEIGHT, bg=BACKGROUND_COLOR)
        self.canvas.grid(row=0,column=0,columnspan=2)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.predict_button = Button(self.root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, padx=(0, 100))

        self.clear_button = Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1)

        self.label = Label(self.root,text="Draw digit for \nneural network.")
        self.label.grid(row=1, column=0, columnspan=3, pady=10, padx=(50, 50))

        self.predict_button.config(font=FONT, width=8, height=1, bg=BUT1_COLOR, fg=FONT_COLOR)
        self.clear_button.config(font=FONT, width=8, height=1, bg=BUT2_COLOR, fg=FONT_COLOR)
        self.label.config(font=FONT, fg=FONT_COLOR)

        # Loading mnist model
        self.model = load_model(MODEL)
        
    # Method that clears canvas
    def clear_canvas(self):
        self.canvas.delete("all")

    # Method that draws on click
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=DRAW_COLOR)
        
    # Method that predicts driven digin
    def predict(self):
        # Updating canvas to get image
        self.canvas.update()
        self.canvas.postscript(file=f"images/{IMAGE_NAME}_from-canvas.eps", colormode='gray')

        # Opening saved from canvas image
        img = Image.open(f"images/{IMAGE_NAME}_from-canvas.eps")
        # Saving it
        img.save(f"images/{IMAGE_NAME}_drawn.jpg")
        # Opening and converting
        img = Image.open(f"images/{IMAGE_NAME}_drawn.jpg").convert('L')
        # Resizing
        img = img.resize((28, 28))
        # To numpy array
        img = np.array(img)
        # From white bg black digit to black bg white digit
        img = cv2.bitwise_not(img)
        # Normalizing from 0 to 1 instead of 0 to 255
        img = img / 255.

        # Reshaping for kmeans (784,)
        X = img.reshape(-1, 1)
        # Fitting kmeans with image
        kmeans = KMeans(n_clusters=3, n_init=3).fit(X)
        # Getting segmented image (only 3 bright collor instead of n)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        # Reshaping for matplotlib
        img = segmented_img.reshape(28, 28)

        # Showing to check preprocessed image
        '''plt.imshow(img, cmap='gray')
        plt.show()'''

        # Saving preprocessed image (what neural network sees)
        image = Image.fromarray((img * 255).astype(np.uint8)).resize((280, 280)).convert('L')
        image.save(f"images/{IMAGE_NAME}_preprocessed.jpg")

        # Predicting with model
        predictions = self.model.predict(img.reshape(1, 28, 28, 1))
        digit = np.argmax(predictions)

        # Printing it in our app
        self.label.config(text="Neural Network\nprediction: "+str(digit))
        self.label.grid(row=1,column=0,columnspan=3,pady=10)
        
# Calling our app
if __name__ == "__main__":
    app = DrawingApplication()
    app.root.mainloop()
