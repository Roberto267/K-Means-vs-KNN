import csv
import math
from collections import Counter

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

# Función para predecir la clase de un punto de prueba
def predict_class(train_data, train_labels, test_point, k):
    # Calcular distancias entre el punto de prueba y todos los puntos de entrenamiento
    distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i]) for i in range(len(train_data))]
    
    # Ordenar las distancias y seleccionar los k vecinos más cercanos
    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_neighbors = sorted_distances[:k]
    
    # Contar las frecuencias de las clases de los vecinos más cercanos
    class_counts = Counter(neighbor[1] for neighbor in k_nearest_neighbors)
    
    # Devolver la clase más común entre los k vecinos más cercanos
    return class_counts.most_common(1)[0][0]

# Lectura del archivo CSV
data = []
labels = []

#with open('diabetes_prediction_dataset.csv', 'r') as file:
with open('diabete.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Ignorar la primera fila (encabezados)
    for row in reader:
        data.append([str(row[0]), float(row[1]), float(row[2]), float(row[3]), str(row[4]), float(row[5]), float(row[6]), float(row[7])])
        labels.append(int(row[8]))

gender_index = header.index('gender')
smoking_index = header.index('smoking_history')

for row in data:
    row[gender_index] = 1 if row[gender_index] == 'male' else 0
    row[smoking_index] = 1 if row[smoking_index] == 'smoker' else 0

# Dividir los datos en entrenamiento y prueba
split_ratio = 0.8
split_index = int(split_ratio * len(data))
train_data, test_data = data[:split_index], data[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Realizar predicciones para cada punto de prueba
k_value = 2
predictions = [predict_class(train_data, train_labels, test_point, k_value) for test_point in test_data]

# Calcular la precisión del modelo
correct_predictions = sum(1 for true_label, predicted_label in zip(test_labels, predictions) if true_label == predicted_label)
accuracy = correct_predictions / len(test_labels)
print(f'Precisión del modelo: {accuracy}')

# Mostrar la clasificación de cada dato en el conjunto de prueba
print("Predicciones:")
print(predictions)