import matplotlib.pyplot  as plt

def plot(img):
    plt.figure()
    plt.imshow(img.squeeze(), cmap='gray')  # 'squeeze()' para eliminar la dimensión de 1
    plt.axis('off')  # No mostrar ejes
    plt.show()

if __name__ == "__main__":
    print("se esta ejecutando aca")