import keras
from keras_preprocessing.image import ImageDataGenerator

def generate_from_derictory(img_path,batch_size,class_mode = None):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.4,
            zoom_range=0.4,
#            featurewise_center=True,
            #samplewise_center= True,
#            featurewise_std_normalization= True,
            #samplewise_std_normalization= True,
#            zca_whitening= True,
            #zca_epsilon=1e-06,
            rotation_range=180,
            width_shift_range=0.4,
            height_shift_range=0.4,
            brightness_range=[0.8,1.2],
            channel_shift_range=10,
#            fill_mode='nearest',
            fill_mode='wrap',
            #cval=0.0,
            horizontal_flip=True,
            vertical_flip= True,
            #validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
            directory=img_path,
            target_size=(400,400),
            batch_size= batch_size,
            save_to_dir= "./new_bx",
            save_prefix= 0,#区别类别
            class_mode= class_mode)
    return train_generator

data_gen = generate_from_derictory('./bx',144,None)
for im in data_gen:
    print("ok")
#for i in range(1000):
#    generate_from_derictory('./bx_115',64,None)
#    #data_gen
#    print(i)
