from utils import *
from sklearn.model_selection import train_test_split
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 4)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

#### STEP 1
path = 'Data'
data = importDataInfo(path)

#### STEP 2
data = balanceData(data, display=True)

#### STEP 3
imagesPath, steerings = loadData(path, data)

del data
del path

#### STEP 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.25, random_state=5)
print('Total Training Images:', len(xTrain))
print('Total Validation Images:', len(xVal))

del imagesPath
del steerings

#### STEP 5


#### STEP 6


#### STEP 7


#### STEP 8
model = createModel()
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#### STEP 9
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=200, epochs=50,
                    validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200, callbacks=[callback])

#### STEP 10
model.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
