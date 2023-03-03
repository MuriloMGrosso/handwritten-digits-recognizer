import tkinter as tk
import numpy as np
import neural_network as neural

class Drawing_Canvas():
    def __init__(self, canvas_size, grid_size):
        self.background_color = '#ffffff'
        self.main_color = '#000000'

        self.canvas = tk.Canvas(root, bg=self.background_color)
        self.canvas.place(width=canvas_size, height=canvas_size)

        self.canvas_per_grid = canvas_size/grid_size
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size * grid_size,1))

        self.canvas.bind('<B1-Motion>',self.draw)
        self.canvas.bind('<Button-1>',self.draw)
        self.canvas.bind('<B3-Motion>',self.erase)
        self.canvas.bind('<Button-3>',self.erase)
    
    def draw(self,event):
        x, y = self.canvas_to_grid(event)
        if x < self.grid_size and y < self.grid_size:
            self.set_pixel(x, y, self.main_color)
            self.grid[int(y * self.grid_size + x)] = [255]
            self.evaluate()

    def erase(self, event):
        x, y = self.canvas_to_grid(event)
        if x < self.grid_size and y < self.grid_size:
            self.set_pixel(x, y, self.background_color)
            self.grid[int(y * self.grid_size + x)] = [0]
            self.evaluate()

    def canvas_to_grid(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = x//self.canvas_per_grid, y//self.canvas_per_grid
        return grid_x, grid_y

    def set_pixel(self,x,y,color):
        fixed_x, fixed_y = x * self.canvas_per_grid, y * self.canvas_per_grid
        self.canvas.create_rectangle(fixed_x, fixed_y, fixed_x + self.canvas_per_grid, fixed_y + self.canvas_per_grid, fill=color, width=0)

    def evaluate(self):
        output = net.evaluate(self.grid)

root = tk.Tk()
root.geometry('308x308')
root.resizable(False, False)
root.title('Neural Network')

net = neural.Network()
net.load('../network_data/DigitsRecognizer.npy')

drawing_canvas = Drawing_Canvas(308,28)

root.mainloop()