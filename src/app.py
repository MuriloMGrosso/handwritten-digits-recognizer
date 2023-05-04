import tkinter as tk
import numpy as np
import neural_network as neural
import math
import colorsys

class Drawing_Canvas():
    def __init__(self, canvas_size, grid_size):
        self.canvas = tk.Canvas(root, bg='#ffffff')
        self.canvas.place(width=canvas_size, height=canvas_size)

        self.canvas_per_grid = canvas_size/grid_size
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size * grid_size,1))

        self.canvas.bind('<B1-Motion>',self.draw)
        self.canvas.bind('<Button-1>',self.draw)
        self.canvas.bind('<B3-Motion>',self.erase)
        self.canvas.bind('<Button-3>',self.erase)
    
    def draw(self,event):
        x, y, dist_center = self.canvas_to_grid(event)
        if x < self.grid_size and y < self.grid_size:
            index = int(y * self.grid_size + x)
            self.grid[index] = np.clip(int(255*(1 - dist_center)) + self.grid[index],0,255)
            self.set_pixel(x, y, hsv_to_hex(0,0,255 - int(self.grid[index][0])))
            evaluate(self.grid)

    def erase(self, event):
        x, y, dist_center = self.canvas_to_grid(event)
        if x < self.grid_size and y < self.grid_size:
            index = int(y * self.grid_size + x)
            self.grid[index] = np.clip(self.grid[index] - int(255*(1 - dist_center)),0,255)
            self.set_pixel(x, y, hsv_to_hex(0,0,255 - int(self.grid[index][0])))
            evaluate(self.grid)

    def clear(self):
        self.canvas.create_rectangle(0, 0, self.grid_size * self.canvas_per_grid,  self.grid_size * self.canvas_per_grid, fill='#ffffff', width=0)
        self.grid = np.zeros((self.grid_size * self.grid_size,1))
        evaluate(self.grid)  

    def fill(self):  
        self.canvas.create_rectangle(0, 0, self.grid_size * self.canvas_per_grid,  self.grid_size * self.canvas_per_grid, fill='#000000', width=0)
        self.grid = np.ones((self.grid_size * self.grid_size,1)) * 255
        evaluate(self.grid)     

    def canvas_to_grid(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = x//self.canvas_per_grid, y//self.canvas_per_grid
        dist_center = math.sqrt((grid_x + 0.5 - x/self.canvas_per_grid)**2 + (grid_y + 0.5 - y/self.canvas_per_grid)**2)
        return grid_x, grid_y, dist_center

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

def hsv_to_hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

root = tk.Tk()
root.geometry('512x512')
root.config(bg='#45ADA8')
root.resizable(False, False)
root.title('Neural Network')

net = neural.Network()
net.load('DigitsRecognizer0.2')

stats_canvas = Stats_Canvas()
drawing_canvas = Drawing_Canvas(308,28)
label = tk.Label(root, text="?", font=("Arial", 128), fg='#E5FCC2', bg='#45ADA8')
label.place(x=100, y=310)

clear_button = tk.Button(root,text="CLEAR",command=drawing_canvas.clear, font=("Arial", 8))
clear_button.pack()
fill = tk.Button(root,text="FILL",command=drawing_canvas.fill, font=("Arial", 8))
fill.pack()

root.mainloop()