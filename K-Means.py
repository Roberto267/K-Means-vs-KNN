import numpy as np
import csv

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def preprocess_data(data):
    headers = data[0]
    data = data[1:]

    # Convertir 'gender' y 'smoking_history' a valores numéricos
    print(headers)
    gender_index = headers.index('gender')
    smoking_index = headers.index('smoking_history')

    for row in data:
        row[gender_index] = 1 if row[gender_index] == 'male' else 0
        row[smoking_index] = 1 if row[smoking_index] == 'smoker' else 0

    # Eliminar la columna 'diabetes'
    diabetes_index = headers.index('diabetes')
    for row in data:
        del row[diabetes_index]

    return np.array(data, dtype=float)

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Función K-Means
def k_means(data, k, max_iterations=100):
    # Inicializar los centroides aleatoriamente
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Calcular las distancias entre cada punto y los centroides, y asignar cada punto al cluster más cercano
        clusters = [[] for _ in range(k)]
        labels = []
        for idx, point in enumerate(data):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            labels.append(cluster)
            clusters[cluster].append(point)

        # Actualizar los centroides
        for i in range(k):
            centroids[i] = np.mean(clusters[i], axis=0)

    return centroids, clusters, labels

# Cargar y preprocesar los datos
filename = 'diabete.csv'
raw_data = load_data(filename)
datos = preprocess_data(raw_data)

# Número de clusters
k = 2

# Ejecutar el algoritmo K-Means
centroides, clusters, etiquetas = k_means(datos, k)

# Imprimir las etiquetas de los clusters
print("Etiquetas de los clusters:")
print(etiquetas)

# Imprimir los resultados
print("\nCentroides:")
print(centroides)
