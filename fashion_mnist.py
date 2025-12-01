# DEEP LEARNING PARA CLASSIFICAÇÃO DE IMAGENS
# Dataset Fashion-MNIST - Zalando Research

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def carregar_dados():
    fashion_mnist = keras.datasets.fashion_mnist
    (imagens_treino, rotulo_treino), (imagens_teste, rotulo_teste) = fashion_mnist.load_data()
    return imagens_treino, rotulo_treino, imagens_teste, rotulo_teste


def exibir_amostras(imagens, rotulos, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagens[i], cmap="gray")
        plt.xlabel(class_names[rotulos[i]])
    plt.show()


def normalizar_dados(imagens_treino, imagens_teste):
    return imagens_treino / 255.0, imagens_teste / 255.0


def criar_modelo():
    modelo = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


def treinar_modelo(modelo, imagens_treino, rotulo_treino):
    return modelo.fit(
        imagens_treino,
        rotulo_treino,
        epochs=10,
        validation_split=0.2,
        batch_size=32
    )


def avaliar_modelo(modelo, imagens_teste, rotulo_teste):
    perda, acuracia = modelo.evaluate(imagens_teste, rotulo_teste)
    print(f"A acurácia no conjunto de teste é: {acuracia:.4f}")


def plotar_curvas(treinamento):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(treinamento.history['accuracy'], label="Treino")
    plt.plot(treinamento.history['val_accuracy'], label="Validação")
    plt.title("Acurácia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(treinamento.history['loss'], label="Treino")
    plt.plot(treinamento.history['val_loss'], label="Validação")
    plt.title("Loss")
    plt.legend()

    plt.show()


def matriz_confusao(rotulo_teste, predicao):
    matriz = confusion_matrix(rotulo_teste, predicao.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de Confusão")
    plt.show()


def main():
    class_names = [
        "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
    ]

    imagens_treino, rotulo_treino, imagens_teste, rotulo_teste = carregar_dados()

    exibir_amostras(imagens_treino, rotulo_treino, class_names)

    imagens_treino, imagens_teste = normalizar_dados(imagens_treino, imagens_teste)

    modelo = criar_modelo()

    treinamento = treinar_modelo(modelo, imagens_treino, rotulo_treino)

    avaliar_modelo(modelo, imagens_teste, rotulo_teste)

    plotar_curvas(treinamento)

    predicao = modelo.predict(imagens_teste)

    matriz_confusao(rotulo_teste, predicao)

    print(classification_report(rotulo_teste, predicao.argmax(axis=1)))


if __name__ == "__main__":
    main()
