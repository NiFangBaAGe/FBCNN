from sklearn.manifold import TSNE
from keras.models import Model, load_model
from keras import backend as K
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import keras
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def load_data():
    x_test, y_test = _, _

    return x_test, y_test

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []

    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

model = _
model = Model(model.input, model.layers[-2].output)
model.summary()
x, y = load_data()
data = model.predict(x, verbose=1)
X_tsne = TSNE(n_components=2, init='pca', random_state=0)
result = X_tsne.fit_transform(data)
scatter(result, y)
plt.savefig(name+'_tsne.png', dpi=120)
plt.show()

