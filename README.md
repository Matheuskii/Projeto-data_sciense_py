# 🧠 Classificação de Imagens com Deep Learning -- Fashion MNIST

Este projeto utiliza **Deep Learning** para classificar imagens do
dataset **Fashion MNIST**, uma coleção de roupas como camisetas, tenis,
calças, bolsas, etc.\
O objetivo é treinar um modelo capaz de identificar corretamente qual
item aparece em cada imagem.

------------------------------------------------------------------------

## 📂 Dataset

**Fashion MNIST** -- Zalando Research\
- 70.000 imagens em tons de cinza (28×28)\
- 60.000 para treino e 10.000 para teste\
- 10 classes de roupas e acessórios

Classes: - T-shirt/Top\
- Trouser\
- Pullover\
- Dress\
- Coat\
- Sandal\
- Shirt\
- Sneaker\
- Bag\
- Ankle Boot

------------------------------------------------------------------------

## 🤖 Modelo de Deep Learning

O projeto usa uma rede neural do tipo **MLP (Multi-Layer Perceptron)**:

-   `Flatten`\
-   `Dense` com 128 neurônios + ReLU\
-   `Dropout (0.2)` para evitar overfitting\
-   `Dense` final com 10 neurônios + Softmax

------------------------------------------------------------------------

## 📈 Resultados

O modelo é treinado e depois avaliado com o conjunto de teste.\
Durante o treinamento, são gerados gráficos de:

-   **Acurácia (treino e validação)**\
-   **Loss (treino e validação)**

Também é exibida:

-   **Matriz de confusão**\
-   **Classification Report** com precisão, recall e f1-score

------------------------------------------------------------------------

## ▶️ Como executar

1.  Clone o repositório:

```{=html}
<!-- -->
```
    git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git

2.  Instale as dependências:

```{=html}
<!-- -->
```
    pip install -r requirements.txt

3.  Execute o script principal:

```{=html}
<!-- -->
```
    python fashion_mnist_classificacao.py

------------------------------------------------------------------------

## 📦 Tecnologias utilizadas

-   Python\
-   TensorFlow / Keras\
-   NumPy\
-   Matplotlib\
-   Scikit-Learn\
-   Scikit-Plot

------------------------------------------------------------------------

## ✨ Sobre o projeto

Este projeto foi desenvolvido como atividade escolar para praticar
conceitos de:

-   Redes Neurais\
-   Classificação de Imagens\
-   Processamento de Dados\
-   Uso de TensorFlow e Keras
"# Projeto-data_sciense_py" 
