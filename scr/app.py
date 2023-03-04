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
        stats_canvas.update_bars(output)

class Stats_Canvas():
    def __init__(self):
        self.width, self.height = 204, 308
        self.canvas = tk.Canvas(root, bg='#000000')
        self.canvas.place(width=self.width, height=self.height, relx=1.0, rely=0.0, anchor='ne')

        num_bars = 10
        self.bars = []
        for i in range(num_bars):
            bar = self.canvas.create_rectangle(0,0,0,0,fill='#555555', width=0)
            self.update_bar(bar,1,i)
        for i in range(num_bars):
            bar = self.canvas.create_rectangle(0,0,0,0,fill='#ffffff', width=0)
            self.update_bar(bar,0,i)
            self.bars.append(bar)

    def update_bar(self, bar, scale, offset):
        border = 16
        bar_height = (self.height - 11 * border) / 10
        bar_width = self.width - 2 * border

        self.canvas.coords(bar, border, offset * (border + bar_height) + border, border + scale * bar_width, (offset + 1) * (border + bar_height))

    def update_bars(self,scales):
        for i in range(len(self.bars)):
            self.update_bar(self.bars[i],scales[i][0],i)

root = tk.Tk()
root.geometry('512x512')
root.resizable(False, False)
root.title('Neural Network')

net = neural.Network()
net.load('../network_data/DigitsRecognizer.npy')

stats_canvas = Stats_Canvas()
drawing_canvas = Drawing_Canvas(308,28)

root.mainloop()