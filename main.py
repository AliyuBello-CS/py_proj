# # Test 1
# import numpy as np

# X = np.array([
#      [1, 5],
#     [2, 10],
#     [4, 20],
#     [5, 25]
# ])

# Y = np.array([0,0,1,1])
# print (X.shape)
# print (Y.shape)

# # Calculate scores using weights and bias.
# # Convert scores into probabilities using sigmoid.
# # Compare those probabilities with the actual outcomes to calculate loss.
# # During training, adjust weights and bias to reduce that loss.

# # EX1

# study_hours = 4
# practice_questions = 5

# weight_1 = 0.5
# weight_2 = 0.1
# bias = -3

# score = (study_hours * weight_1) + (practice_questions * weight_2) + bias

# print(score)


# # EX2
# probability = 1 / (1 + np.exp(-score))

# print("Score:", score)
# print("Probability of passing:", probability)
# if probability >= 0.5:
#     prediction = 1
# else:
#     prediction = 0

# print("Prediction:", prediction)


# # EX3
# weights = np.array([0.5, 0.1])
# bias = -3

# scores2 = X @ weights + bias

# print(scores2)

# probabilities = 1 / (1 + np.exp(-scores2))

# print(probabilities)

# predictions = (probabilities >= 0.5).astype(int)


# errors = probabilities - y

# study_hours = X[:, 0]
# contributions = study_hours * errors

# study_weight_gradient = np.mean(contributions)

# print("Contributions:", contributions)
# print("Study-hours gradient:", study_weight_gradient)
# print(predictions)

# # EX4= PROBA AND PRED loss


# probability = 0.8
# loss = -np.log(probability)
# print (loss)


# probability = 0.2
# loss = -np.log(1 - probability)

# print(loss)


# # CROSS ENTROPY

# y = np.array([0, 0, 1, 1])

# losses = -(y * np.log(probabilities)
#            + (1 - y) * np.log(1 - probabilities))

# print("Each student's loss:", losses)

# loss = np.mean(losses)

# print("Average loss:", loss)

# # cont

# # GRADIENT

# errors = probabilities - y

# study_hours = X[:, 0]
# contributions = study_hours * errors

# study_weight_gradient = np.mean(contributions)

# print("Contributions:", contributions)
# print("Study-hours gradient:", study_weight_gradient)
# //

import numpy as np

# 1. Training data
# Columns: study hours, practice questions completed
X = np.array([
    [1, 5],
    [2, 10],
    [4, 20],
    [5, 25]
], dtype=float)

# Actual outcomes: 0 = failed, 1 = passed
y = np.array([0, 0, 1, 1])

# 2. Starting parameters
weights = np.array([0.5, 0.1])
bias = -3.0

learning_rate = 0.01
n_steps = 100

# 3. Measure loss before training
scores = X @ weights + bias

# Binary cross-entropy calculated safely from scores
initial_loss = np.mean(np.logaddexp(0, scores) - y * scores)

print(f"Initial loss: {initial_loss:.4f}")

# 4. Train: repeatedly adjust the weights and bias
for step in range(n_steps):

    # Predict probabilities using the current parameters
    scores = X @ weights + bias
    probabilities = 1 / (1 + np.exp(-scores))

    # Calculate gradients
    errors = probabilities - y
    weight_gradients = (X.T @ errors) / len(y)
    bias_gradient = np.mean(errors)

    # Update parameters
    weights = weights - learning_rate * weight_gradients
    bias = bias - learning_rate * bias_gradient

    # Check loss after every 20 updates
    if (step + 1) % 20 == 0:
        updated_scores = X @ weights + bias
        current_loss = np.mean(
            np.logaddexp(0, updated_scores) - y * updated_scores
        )

        print(f"Step {step + 1}: loss = {current_loss:.4f}")

# 5. Make predictions using the final learned parameters
final_scores = X @ weights + bias
final_probabilities = 1 / (1 + np.exp(-final_scores))
final_predictions = (final_probabilities >= 0.5).astype(int)

final_loss = np.mean(
    np.logaddexp(0, final_scores) - y * final_scores
)

accuracy = np.mean(final_predictions == y)

# 6. Display the results
print("\n--- Results after training ---")
print("Learned weights:", weights)
print("Learned bias:", bias)
print("Probabilities:", np.round(final_probabilities, 3))
print("Predictions:", final_predictions)
print("Actual outcomes:", y)

print(f"Initial loss: {initial_loss:.4f}")
print(f"Final loss:   {final_loss:.4f}")
print(f"Training accuracy: {accuracy:.1%}")