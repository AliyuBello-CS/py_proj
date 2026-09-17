# Test 1
import numpy as np

X = np.array([
     [1, 5],
    [2, 10],
    [4, 20],
    [5, 25]
])

Y = np.array([0,0,1,1])
print (X.shape)
print (Y.shape)


# EX1

study_hours = 4
practice_questions = 5

weight_1 = 0.5
weight_2 = 0.1
bias = -3

score = (study_hours * weight_1) + (practice_questions * weight_2) + bias

print(score)


# EX2
probability = 1 / (1 + np.exp(-score))

print("Score:", score)
print("Probability of passing:", probability)
if probability >= 0.5:
    prediction = 1
else:
    prediction = 0

print("Prediction:", prediction)


# EX2
weights = np.array([0.5, 0.1])
bias = -3

scores2 = X @ weights + bias

print(scores2)

probabilities = 1 / (1 + np.exp(-scores2))

print(probabilities)

predictions = (probabilities >= 0.5).astype(int)

print(predictions)