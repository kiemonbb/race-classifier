import matplotlib.pyplot as plt
import argparse
import yaml
from tensorflow.keras import optimizers
from models.model_factory import get_model
from dataset import get_generators


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    model = get_model(
        model_name=config["model_name"],
        input_shape=(image_size[0], image_size[1], 3)
    )

    train_generator, val_generator = get_generators(image_size, batch_size)

    model.compile(optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(train_generator, epochs=epochs,
                     validation_data=val_generator, steps_per_epoch=2, validation_steps=2)

    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])
    plt.show()


if __name__ == "__main__":
    main()
