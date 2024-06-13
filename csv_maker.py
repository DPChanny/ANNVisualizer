import tkinter as tk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")

c = ["red", "blue", "green"]
cc = 0

df.info()
df.head()

root = tk.Tk()

fig = plt.Figure(figsize=(10, 10))
canvas = FigureCanvasTkAgg(fig, root)


def on_click(event):
    global cc, c
    print(event)
    print(canvas.get_width_height())
    df.loc[len(df)] = [event.x / canvas.get_width_height()[0], 1 - event.y / canvas.get_width_height()[1], c[cc]]
    draw()


def on_class(event):
    global cc, c
    cc += 1
    if cc == len(c):
        cc = 0
    print(cc)


def draw():
    fig.clf()
    ax = fig.add_subplot()
    ax.axis('off')

    x = df[['x', 'y']].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['class'].values)
    for i, j in enumerate(np.unique(y)):
        ax.scatter(x[y == j, 0], x[y == j, 1], s=5, color=ListedColormap(('red', 'green', 'blue'))(i))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def on_save(event):
    df.to_csv("data.csv")


draw()

root.bind("<Button-1>", on_click)
root.bind("<Button-2>", on_class)
root.bind("<Button-3>", on_save)

root.mainloop()
