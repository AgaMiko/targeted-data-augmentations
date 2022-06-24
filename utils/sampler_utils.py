def create_weights_vector(labels, weights):
    weighted_vector = list()
    for label in labels:
        weighted_vector.append(weights[label])
    return weighted_vector