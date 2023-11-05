import math
import tensorflow as tf
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt


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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(
    x_train,
    y_train,
    epochs=1,
)

#model.save_weights('./model/fgsm_mnist')
#model.load_weights('./model/fgsm_mnist')

def getAccuracy(model, x_test, y_test):
    predictions = model.predict(x_test, verbose=1)
    print(np.argmax(predictions[0]))
    right = 0
    for p in range(len(predictions)):
        if np.argmax(predictions[p]) == y_test[p]:
            right+=1

    return right/len(predictions)

def l2_norm(img):
    return math.sqrt(np.sum(np.square(img)))

def l_inf(img):
    return np.max(np.abs(img))

def showImage(img, save=False, img_name='', trackNorm=False, norm=0.0, inf_norm=0.0):
    plt.imshow(img, cmap='gray')
    if trackNorm:
        plt.text(0,-2,'L2-Norm:   '+str(norm))
        plt.text(0,-4,'Linf-Norm: '+str(inf_norm))
    if save:
        plt.savefig(img_name)
        plt.close()
    else:
        plt.show()


def return_loss(y_predict, y_true):
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    return loss_func(y_true, y_predict)



def create_adversarial_pattern(input_image, target):
    _input_image = tf.Variable(input_image, trainable=True)

    with tf.GradientTape(watch_accessed_variables=True) as g: #gradientTape to calculate gradient (dg(x))
        g.watch(_input_image) #init input_image as value for x
        prediction = model(_input_image)
        one_hot = tf.one_hot(target, 10)
        one_hot = tf.reshape(one_hot, (10000, 10))
        loss = return_loss(prediction, one_hot)

    grad = g.gradient(loss, _input_image)
    return grad

def layerGradientOnImage(input_image, gradient, epsilon):
    grad_signs = tf.sign(gradient)
    adv_inputs = epsilon*tf.keras.backend.get_value(grad_signs)

    return input_image + adv_inputs


epsilon = [x/20 for x in range(21)]
fgsm_accuracy = [] # GOAL: calculate fgsm_accuracy(epsilon)

print(f"Accuracy: {getAccuracy(model, x_test, y_test)*100}%")

#save already generated adversial testing_images to use them
#for re-training the model and adapting to the fgsm-attack
epsilon_limit = float(input('Define epsilon_limit (maximum e in e*g\'(x) the model will be re-trained with): '))    
adversial_patterns = []
adversial_y = []

#run FGSM for test set; get average accuracy
gradient=create_adversarial_pattern(x_test, y_test)

for i in epsilon:
    adv_x = layerGradientOnImage(x_test, gradient, i)
    if i <= epsilon_limit:
        adversial_patterns.extend(adv_x)
        adversial_y.extend(y_test)
    showImage(adv_x[0], save=True, img_name=f'fgsm-{i}.png', trackNorm=True, norm=l2_norm(adv_x[0]-x_test[0]), inf_norm=l_inf(adv_x[0]-x_test[0]))

    curr_acc = getAccuracy(model, adv_x, y_test)*100
    print(f"Accuracy under FGSM with epsilon {i}: {curr_acc}%")
    fgsm_accuracy.append(curr_acc)

print(np.array(adversial_patterns).shape)
print(np.array(adversial_y).shape)
plt.plot(epsilon, fgsm_accuracy)
plt.xlim(xmin=0.0)
plt.ylim(ymin=0.0)
plt.savefig(f"initial_fgsm_of_e-{epsilon_limit}.png")
plt.close()

#Retraining model with epsilon 
model.fit(
    x=np.array(adversial_patterns),
    y=np.array(adversial_y),
    epochs=1
)

retrained_accuracy = getAccuracy(model, x_test, y_test)
print(f"Retrained Accuracy: {retrained_accuracy}%")


new_adv_set = x_train[:10000]
adv_y_train = y_train[:10000]

new_adv_set_fgsm_accuracy = []
for i in epsilon:
    layered_adv_set = layerGradientOnImage(new_adv_set, gradient, i)
    curr_acc = getAccuracy(model, layered_adv_set, adv_y_train)*100
    print(f"Retrained Accuracy under fgsm with epsilon {i}: {curr_acc}%")
    new_adv_set_fgsm_accuracy.append(curr_acc)

plt.plot(epsilon, new_adv_set_fgsm_accuracy)
plt.xlim(xmin=0.0)
plt.ylim(ymin=0.0)
plt.savefig(f"retrained_fgsm_of_e-{epsilon_limit}.png")
plt.close()


#use gradient of current 
new_adv_set = x_train[:10000]
adv_y_train = y_train[:10000]

new_adv_set_fgsm_accuracy = []
gradient = create_adversarial_pattern(x_test, y_test)

for i in epsilon:
    layered_adv_set = layerGradientOnImage(new_adv_set, gradient, i)
    curr_acc = getAccuracy(model, layered_adv_set, adv_y_train)*100
    print(f"Retrained Accuracy under fgsm with epsilon {i}: {curr_acc}%")
    new_adv_set_fgsm_accuracy.append(curr_acc)

plt.plot(epsilon, new_adv_set_fgsm_accuracy)
plt.xlim(xmin=0.0)
plt.ylim(ymin=0.0)
plt.savefig(f"retrained_fgsm_of_e_current_gradient-{epsilon_limit}.png")
plt.close()
