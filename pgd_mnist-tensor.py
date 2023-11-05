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
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(
    x_train,
    y_train,
    epochs=1,
)

#model.save_weights('./model/pgd_mnist')
#model.load_weights('./model/pgd_mnist')

def getAccuracy(model, x_test, y_test):
    predictions = model.predict(x_test, verbose=1)
    print(np.argmax(predictions[5]))
    right = 0
    for p in range(len(predictions)):
        if np.argmax(predictions[p]) == y_test[p]:
            right+=1

    return right/len(predictions)


def showImage(img, save=False, img_name='', trackNorm=False, norm=0.0, inf_norm=0.0):
    plt.imshow(img, cmap='gray')
    if trackNorm:
        plt.text(0,-2,'L2-Norm:   '+str(norm))
        plt.text(0,-4,'Linf-Norm: '+str(inf_norm))
    if save:
        plt.savefig(img_name)
    plt.show()


def return_loss(y_predict, y_true):
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    return loss_func(y_true, y_predict)



def getGradient(input_image, target):
    _input_image = tf.Variable(input_image, trainable=True)

    with tf.GradientTape(watch_accessed_variables=True) as g: #gradientTape to calculate gradient (dg(x))
        g.watch(_input_image) #init input_image as value for x
        prediction = model(_input_image)
        one_hot = tf.one_hot(target, 10)
        one_hot = tf.reshape(one_hot, (1, 10))
        loss = return_loss(prediction, one_hot)

    grad = g.gradient(loss, _input_image)
    return grad


def l2_norm(img):
    return math.sqrt(np.sum(np.square(img)))

def l_inf(img):
    return np.max(np.abs(img))

def isInsideBx(adv_grad, max_norm=20):
    current_diff = math.sqrt(np.sum(np.square(np.array(adv_grad))))
    return current_diff < max_norm

def isWrongPredict(adv_image, target):
    return np.argmax(model.predict(np.array(adv_image), verbose=0)[0]) != target



#input_image: 3D list of shape (1, 28, 28)
#target: y of input_image; the correct value corresponding to the input_image
#alpha: the factor the gradient gets applied on the image on each iteration
#max_iterations: maximum number of iterations (T)
#max_norm: the maximum distance from input_image (x0) to the changed image (x*) measured as l2-distance
def createAdversialPattern(input_image, target, alpha=0.01, max_iterations=50, max_norm=5):

    for i in range(max_iterations):
        gradient = getGradient(input_image, target)
        grad_signs = tf.sign(gradient)

        adv_grad = alpha*tf.keras.backend.get_value(grad_signs)

        if not isInsideBx(adv_grad, max_norm=max_norm):
            break
        input_image += adv_grad
        if isWrongPredict(input_image, target):
            break
    
    return input_image[0]


##TEST
# showImage(x_test[0], save=True, img_name='pgd_initial_image.png')
# print(y_test[0])

adv_pattern = [createAdversialPattern([x_test[i]], y_test[i]) for i in range(len(x_test[:100]))]
xd = adv_pattern[0] - x_test[:100]
predictions = model.predict(np.array(adv_pattern))
predictions = np.array([np.argmax(x) for x in predictions])


print(np.sum(predictions == y_test[:100])/len(predictions))

showImage(adv_pattern[0], save=True, img_name='pgd_adversarial_image.png', trackNorm=True, norm=l2_norm(np.array(adv_pattern[0] - x_test[0])), inf_norm=l_inf(np.array(adv_pattern[0]-x_test[0])))
showImage(adv_pattern[0] - x_test[0], save=True, img_name='pgd_adversial_pattern.png', trackNorm=True, norm=l2_norm(np.array(adv_pattern[0] - x_test[0])), inf_norm=l_inf(np.array(adv_pattern[0]-x_test[0])))
print(np.argmax(model.predict(np.array([adv_pattern[0]]))))
