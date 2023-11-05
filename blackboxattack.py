import tensorflow as tf
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import math

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(
    x_train,
    y_train,
    epochs=1,
)

def getAccuracy(model, x_test, y_test):
    predictions = model.predict(x_test, verbose=0)
    print(np.argmax(predictions[5]))
    right = 0
    for p in range(len(predictions)):
        if np.argmax(predictions[p]) == y_test[p]:
            right+=1

    return right/len(predictions)


def showImage(img, show=True, save=False, img_name='', trackNorm=False, norm=0.0, inf_norm=0.0):
    plt.imshow(img, cmap='gray')
    if trackNorm:
        plt.text(0,-2,'L2-Norm:   '+str(norm))
        plt.text(0,-4,'Linf-Norm: '+str(inf_norm))
    if save:
        plt.savefig(img_name)
    if show:
        plt.show()
    else:
        plt.close()



def getPrediction(model, input_image):
    return np.argmax(model.predict(np.array([input_image]), verbose=0))

def getVectorLength(vec):
    return math.sqrt(np.sum(np.square(vec)))


def coarseGrainedSearch(model, input_image, direction, v, target, alpha=0.1):
    left, right = v, v

    if np.argmax(model.predict(np.array([input_image + direction * v]), verbose=0)[0]) == target:
        right+=alpha
        while np.argmax(model.predict(np.array([input_image + direction * right]), verbose=0)[0]) == target:
            if np.max(model.predict(np.array([input_image + direction * right]), verbose=0)[0]) == 1:
                print(1.0)
            right*=(1+alpha)
    else:
        left-=alpha
        while np.argmax(model.predict(np.array([input_image + direction * left]), verbose=0)[0]) != target:
            if np.max(model.predict(np.array([input_image + direction * right]), verbose=0)[0]) == 1:
                print(1.0)
            left*=(1-alpha)
    return (left, right)

def binarySearch(model, input_image, direction, left, right, target, max_diff=0.005):
    print('BINARY')
    while right - left > max_diff:
        mid = (right + left) / 2
        if np.argmax(model.predict(np.array([input_image + direction * mid]), verbose=0)[0]) == target:
            left = mid
        else:
            right = mid
    return right

def blackboxattack(model, input_image, target, T=20, alpha=0.001, max_diff=0.005, b=0.005):
    #init random direction vector
    direction = np.random.rand(28, 28) - 0.5
    v_prev = 1
    for t in range(T):
        direction /= getVectorLength(direction)
        print(f'Number: {t}')
        #calculate h(s)
        (vleft, vright) = coarseGrainedSearch(model, input_image[0], direction, v_prev, target)
        hs = binarySearch(model, input_image[0], direction, vleft, vright, target)

        showImage(input_image[0] + hs*direction/getVectorLength(direction))
        print(f'V1: {hs}')
        
        #calculate h(s + b*u)

        #generate zero mean gaussian distributed vector (u)
        u = np.random.normal(0, 1, size=(28, 28))

        (uleft, uright) = coarseGrainedSearch(model, input_image[0], direction, v_prev, target)
        hsbu = binarySearch(model, input_image[0], direction + b*u, uleft, uright, target)

        v_prev = hs

        g = (hsbu-hs) * u / b
        print(f'V2: {hsbu}')
        print(f'Length: {getVectorLength(g)}')
        print(np.argmax(np.abs(g)))
        direction -= alpha * g

    
    hleft, hright = coarseGrainedSearch(model, input_image[0], direction, v_prev, target)
    h = binarySearch(model, input_image[0], direction, hleft, hright, target)

    return input_image[0] + h*direction/getVectorLength(direction)


        


adv_image = blackboxattack(model, x_test[0:1], y_test[0])
showImage(adv_image, show=True, save=True, img_name='bba_rgf_2', trackNorm=True, norm=getVectorLength(adv_image - x_test[0]), inf_norm=np.max(np.abs(adv_image-x_test[0])))
