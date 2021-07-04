 #contruir el modelo CCN
# recordar las carpetas deben estar asi como estan para que keras las pueda leer 

#  primero las librerias
from keras.models import Sequential #inicializacion de la red neuronal
from keras.layers import Convolution2D #primera parte de una red neuronal convolucional ... esta libreria realiza la capa de convolucion
from keras.layers import MaxPooling2D # realzan la capa de Max Poolin
from keras.layers import Flatten #para pasar todo a un array 
from keras.layers import Dense # para unir 


#inicializamos la CCN
classifier = Sequential()

# capa de convolucion
# se trata de tomar la imagen , y se le aplica una especie de "filtros" con "detector de Rasgos" (de matriz mas chica) y que multiplica a la original
# se va deslizando desde la izquierda a la derecha multiplicando , esto nos da como resultado un "mapa de caracteristicas" (son varios mapas de caracteristicas)
# Convolution2D :
# kernel_size es el tama'o del filtro , puede ser una matriz 3x3
# filtros= es la cantidad de "mapas de caracteristicas" que queremos , si ponemos 32 , tendremos 32 "mapas de caracteristicas" filtradas por una matriz 3x3
# si la imagen no es cuadrada recordar poner el input_shape
# activation= funcion de activacion
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(64,64,3),activation='relu')) 


# max pooling
# se le aplica el max pooling al mapa de caracteristicas, y me queda un mapa de caracteristicas pooled
# se toma el numero mas grande de la matriz filtro (pool_size) que va a iterar por el mapa de caracteristicas
# pool_size= la ventana que va a ir filtrando el mapa de caracteristicas , normalmente se utiliza un 2x2 
classifier.add(MaxPooling2D(pool_size=(2,2)))

#puedo repertir la operacion para reducir la info 2 veces, y asi entrenar una red neuronal mas potente
#esta red va a tomar mas detalles como la oreja del gato , etc
#input_shape=(64,64,3) lo elimino
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))



# flatting
classifier.add(Flatten())


#Creamos la red neuronal
# YA tenemos las imagenes en flatten ... ahora listo para pasar por la red neuronal
classifier.add(Dense(units=128,activation='relu'))

# units pongo 1 ya que es binario , o es perro o gato
classifier.add(Dense(units=1,activation='sigmoid'))


# compilar la red neuronal de convolucion
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# ajustaremos la red neuronal a las imagenes para entrenar

from keras.preprocessing.image import ImageDataGenerator
train_datagen= ImageDataGenerator(
    rescale=1./255, #me transforma los pixeles de 0 a 1 y no de 0 a 255 como era
    shear_range=0.2, #porrcentaje
    zoom_range=0.2, # porcentaje de zoom
    horizontal_flip=True)


test_datagen= ImageDataGenerator(rescale=1./255) # hago el escalado pero ahora para el test

training_dataset = train_datagen.flow_from_directory(
                                                    'dataset/training_set',
                                                    target_size=(64,64), #el tama;no de las imagenes
                                                    batch_size=32, #el tama;o del bloque de carga 
                                                    class_mode='binary') #aca solo por que tengo dos categorias , sino se deberia cambiar

testing_dataset= test_datagen.flow_from_directory(
                                                    'dataset/test_set',
                                                    target_size=(64,64), #el tama;no de las imagenes
                                                    batch_size=32, #el tama;o del bloque de carga 
                                                    class_mode='binary') #aca solo por que tengo dos categorias , sino se deberia cambiar

classifier.fit_generator(training_dataset, #primero el conjunto de entrenamiento
                        steps_per_epoch=8000,
                        epochs=50,
                        validation_data= testing_dataset, #lo valido con el test
                        validation_steps=2000)
