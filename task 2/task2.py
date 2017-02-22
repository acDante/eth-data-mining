import random
import numpy as np

# Constants used throughout the code.
D = 400  # Number of features of every image 
N = 5000 # Number of images in every call of the mapper
epochs = 80 # Number of epochs 

alpha = 0.001 # stepsize

# beta1 and beta2 exponential decay rates for the moment estimates
beta1 = 0.9
beta2 = 0.999

epsilon = 10**(-8)

samples = 2000 # Number of samples
# Creation of the omega and beta samples for Random Fourier Features with Laplacian kernel
# Drawing omega_samples from the Cauchy distribution
omega_samples = np.random.standard_cauchy((samples, D))
beta_samples = np.random.uniform(low=0.0, high=2.0*np.pi, size=samples)

def transform(X):
    # Random Fourier Features with Laplacian kernel
    X_transformed = np.sqrt(2.0/samples)*np.cos(np.dot(X, omega_samples.T) + beta_samples)

    # Add one dimension to the data with a value of 1
    if len(np.shape(X_transformed)) == 1:
        return np.append(X_transformed, [1])
    else:
        shape = np.shape(X_transformed)
        out = np.zeros((shape[0], shape[1] + 1))
        for i in range(shape[0]):
            out[i] = np.append(X_transformed[i], [1])
        return out


def mapper(key, value):
    # key: None
    # value: a Numpy array of arrays. Each mapper gets 5000 lines.
    # The matrix represents the 400 features of the 5000 images provided in every mapper call
    matrix = transform(np.array([image.split()[1:] for image in value]).astype(float))
    new_shape = matrix.shape[1]
    # Label for each image
    y = np.array([float(image.split()[0]) for image in value])
    # Initialization of the weight vector
    w = np.zeros(new_shape)
    # Initialization of the first moment vector
    moment_vector_1 = np.zeros(new_shape)
    # Initialization of the second moment vector
    moment_vector_2 = np.zeros(new_shape)
    # Perform ADAM algorithm
    indexes = range(N)
    for e in range(epochs):
        random.shuffle(indexes)
        timestep = 0
        for t in range(N):
            timestep += 1
            index = indexes[t]
            # Get the gradient of the hinge loss function
            if y[index]*np.inner(w, matrix[index]) < 1:
                gradient = -y[index]*matrix[index]
            else:
                gradient = 0
            moment_vector_1 = beta1*moment_vector_1 + (1.0 - beta1)*gradient # Update biased first moment estimate
            moment_vector_2 = beta2*moment_vector_2 + (1.0 - beta2)*(gradient**2) # Update biased second raw moment estimate
            moment_vector_1_corr = moment_vector_1/(1.0 - beta1**timestep) # Compute biased-corrected first moment estimate
            moment_vector_2_corr = moment_vector_2/(1.0 - beta2**timestep) # Compute bias-corrected second raw moment estimate
            w = w - alpha*moment_vector_1_corr/(np.sqrt(moment_vector_2_corr) + epsilon) # Update weight vector
    yield 0, w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    w = 0
    for value in values:
        w += value

    yield w/float(len(values)) # Output the average of the weight vectors

