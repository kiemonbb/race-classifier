from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pandas as pd
import os


def get_generators(image_size=(224, 224), batch_size=32):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv_path = os.path.abspath(os.path.join(
        current_dir, "..", "data", "train_labels.csv"))
    val_csv_path = os.path.abspath(os.path.join(
        current_dir, "..", "data", "val_labels.csv"))

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    images_dir = os.path.abspath(os.path.join(
        current_dir, "..", "data"))

    train_generator = train_datagen.flow_from_dataframe(train_df,
                                                        directory=images_dir,
                                                        x_col="file",
                                                        y_col="race",
                                                        target_size=image_size,
                                                        batch_size=batch_size,
                                                        class_mode="categorical",
                                                        shuffle=True)
    val_generator = val_datagen.flow_from_dataframe(val_df,
                                                    directory=images_dir,
                                                    x_col="file",
                                                    y_col="race",
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    shuffle=True)
    return train_generator, val_generator
