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

# Calculate scores using weights and bias.
# Convert scores into probabilities using sigmoid.
# Compare those probabilities with the actual outcomes to calculate loss.
# During training, adjust weights and bias to reduce that loss.

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


# EX3
weights = np.array([0.5, 0.1])
bias = -3

scores2 = X @ weights + bias

print(scores2)

probabilities = 1 / (1 + np.exp(-scores2))

print(probabilities)

predictions = (probabilities >= 0.5).astype(int)

print(predictions)

# EX4= PROBA AND PRED loss


probability = 0.8
loss = -np.log(probability)
print (loss)


probability = 0.2
loss = -np.log(1 - probability)

print(loss)


# CROSS ENTROPY

y = np.array([0, 0, 1, 1])

losses = -(y * np.log(probabilities)
           + (1 - y) * np.log(1 - probabilities))

print("Each student's loss:", losses)

loss = np.mean(losses)

print("Average loss:", loss)