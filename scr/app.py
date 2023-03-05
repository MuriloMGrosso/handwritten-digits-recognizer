import tkinter as tk
import numpy as np
import neural_network as neural

class Drawing_Canvas():
    def __init__(self, canvas_size, grid_size):
        self.background_color = '#E5FCC2'
        self.main_color = '#111111'

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
            evaluate(self.grid)

    def erase(self, event):
        x, y = self.canvas_to_grid(event)
        if x < self.grid_size and y < self.grid_size:
            self.set_pixel(x, y, self.background_color)
            self.grid[int(y * self.grid_size + x)] = [0]
            evaluate(self.grid)

    def canvas_to_grid(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = x//self.canvas_per_grid, y//self.canvas_per_grid
        return grid_x, grid_y

    def set_pixel(self,x,y,color):
        fixed_x, fixed_y = x * self.canvas_per_grid, y * self.canvas_per_grid
        self.canvas.create_rectangle(fixed_x, fixed_y, fixed_x + self.canvas_per_grid, fixed_y + self.canvas_per_grid, fill=color, width=0)

class Bar():
    def __init__(self, canvas, color, bg_color, border, canvas_height, canvas_width, offset):
        self.canvas = canvas
        label_space = 32
        self.width = canvas_width - 2*border - label_space
        height = (canvas_height - border) / 10 - border

        self.x0 = border + label_space
        self.y0 = offset * (border + height) + border
        self.x1 = border + label_space + self.width
        self.y1 = (offset + 1) * (border + height)

        self.canvas.create_rectangle(self.x0,self.y0,self.x1,self.y1,fill=bg_color, width=0)
        self.bar = self.canvas.create_rectangle(self.x0,self.y0,self.x1 - self.width,self.y1,fill=color, width=0)
        self.text = self.canvas.create_text((self.x0 + self.x1)/2, (self.y0 + self.y1)/2, text='0%', fill='#E5FCC2', font=("Arial", 16))
        self.canvas.create_text(self.x0/2, (self.y0 + self.y1)/2, text=f'{offset}', fill='#E5FCC2', font=("Arial", 16))

    def update(self, scale):
        self.canvas.coords(self.bar, self.x0, self.y0, self.x1 - (1 - scale) * self.width, self.y1)
        self.canvas.itemconfig(self.text, text=f'{round(scale*100)}%')

class Stats_Canvas():
    def __init__(self):
        self.width, self.height = 204, 512
        self.canvas = tk.Canvas(root, bg='#111111')
        self.canvas.place(width=self.width, height=self.height, relx=1.0, rely=0.0, anchor='ne')

        num_bars = 10
        self.bars = []
        for i in range(num_bars):
            self.bars.append(Bar(self.canvas, '#547980', '#594F4F', 16, self.height, self.width, i))

    def update_bars(self,scales):
        for i in range(len(self.bars)):
            self.bars[i].update(scales[i][0])

def evaluate(data):
    output = net.evaluate(data)
    stats_canvas.update_bars(output)
    label.config(text=f"{np.argmax(output)}")

root = tk.Tk()
root.geometry('512x512')
root.config(bg='#45ADA8')
root.resizable(False, False)
root.title('Neural Network')

net = neural.Network()
net.load('../network_data/DigitsRecognizer.npy')

stats_canvas = Stats_Canvas()
drawing_canvas = Drawing_Canvas(308,28)
label = tk.Label(root, text="?", font=("Arial", 128), fg='#E5FCC2', bg='#45ADA8')
label.place(x=100, y=310)

root.mainloop()