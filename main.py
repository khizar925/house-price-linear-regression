
# Libraries Improt
import numpy as np
import pandas as pd

# Data Import

data = pd.read_csv('Housing.csv')
price = np.array(data['price'])
area = np.array(data['area'])
bedrooms = np.array(data['bedrooms'])
bathrooms = np.array(data['bathrooms'])
stories = np.array(data['stories'])
mainroad = np.array(data['mainroad'])
guestroom = np.array(data['guestroom'])
basement = np.array(data['basement'])
hotwaterheating = np.array(data['hotwaterheating'])
airconditioning = np.array(data['airconditioning'])
parking = np.array(data['parking'])
prefarea = np.array(data['prefarea'])
furnishing_semi_furnished = np.array(data['furnishing_semi_furnished'])
furnishing_unfurnished = np.array(data['furnishing_unfurnished'])

# Train/Test Split
np.random.seed(42)
m = len(price)
indices = np.random.permutation(m)
train_size = int(0.8 * m)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

price_train = price[train_indices]
area_train = area[train_indices]
bedrooms_train = bedrooms[train_indices]
bathrooms_train = bathrooms[train_indices]
stories_train = stories[train_indices]
mainroad_train = mainroad[train_indices]
guestroom_train = guestroom[train_indices]
basement_train = basement[train_indices]
hotwaterheating_train = hotwaterheating[train_indices]
airconditioning_train = airconditioning[train_indices]
parking_train = parking[train_indices]
prefarea_train = prefarea[train_indices]
furnishing_semi_train = furnishing_semi_furnished[train_indices]
furnishing_unfurnished_train = furnishing_unfurnished[train_indices]

price_test = price[test_indices]
area_test = area[test_indices]
bedrooms_test = bedrooms[test_indices]
bathrooms_test = bathrooms[test_indices]
stories_test = stories[test_indices]
mainroad_test = mainroad[test_indices]
guestroom_test = guestroom[test_indices]
basement_test = basement[test_indices]
hotwaterheating_test = hotwaterheating[test_indices]
airconditioning_test = airconditioning[test_indices]
parking_test = parking[test_indices]
prefarea_test = prefarea[test_indices]
furnishing_semi_test = furnishing_semi_furnished[test_indices]
furnishing_unfurnished_test = furnishing_unfurnished[test_indices]

# Feature Scaling (Min-Max)
price_min, price_max = np.min(price_train), np.max(price_train)
price_train_scaled = (price_train - price_min) / (price_max - price_min)
price_test_scaled = (price_test - price_min) / (price_max - price_min)

area_min, area_max = np.min(area_train), np.max(area_train)
area_train_scaled = (area_train - area_min) / (area_max - area_min)
area_test_scaled = (area_test - area_min) / (area_max - area_min)

bedrooms_min, bedrooms_max = np.min(bedrooms_train), np.max(bedrooms_train)
bedrooms_train_scaled = (bedrooms_train - bedrooms_min) / (bedrooms_max - bedrooms_min)
bedrooms_test_scaled = (bedrooms_test - bedrooms_min) / (bedrooms_max - bedrooms_min)

bathrooms_min, bathrooms_max = np.min(bathrooms_train), np.max(bathrooms_train)
bathrooms_train_scaled = (bathrooms_train - bathrooms_min) / (bathrooms_max - bathrooms_min)
bathrooms_test_scaled = (bathrooms_test - bathrooms_min) / (bathrooms_max - bathrooms_min)

stories_min, stories_max = np.min(stories_train), np.max(stories_train)
stories_train_scaled = (stories_train - stories_min) / (stories_max - stories_min)
stories_test_scaled = (stories_test - stories_min) / (stories_max - stories_min)

parking_min, parking_max = np.min(parking_train), np.max(parking_train)
parking_train_scaled = (parking_train - parking_min) / (parking_max - parking_min)
parking_test_scaled = (parking_test - parking_min) / (parking_max - parking_min)

def h_of_x(theta, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishing_semi_furnished,furnishing_unfurnished ):
    return theta[0] + theta[1]*area + theta[2]*bedrooms + theta[3]*bathrooms + theta[4]* stories + theta[5]*mainroad + theta[6]*guestroom + theta[7]*basement + theta[8]*hotwaterheating + theta[9]*airconditioning + theta[10]*parking + theta[11]*prefarea + theta[12]*furnishing_semi_furnished + theta[13]*furnishing_unfurnished

def cost(m, predictions, actual):
  return (1/(2*m)) * np.sum((predictions - actual)**2)

def gradient_descent(theta, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishing_semi_furnished, furnishing_unfurnished, alpha, predictions, actual, m):

    # Create a copy of theta to store updates
    new_theta = theta.copy()

    # Calculate error
    error = predictions - actual

    # Update each theta
    new_theta[0] = theta[0] - alpha * (1/m) * np.sum(error)  # bias term
    new_theta[1] = theta[1] - alpha * (1/m) * np.sum(error * area)
    new_theta[2] = theta[2] - alpha * (1/m) * np.sum(error * bedrooms)
    new_theta[3] = theta[3] - alpha * (1/m) * np.sum(error * bathrooms)
    new_theta[4] = theta[4] - alpha * (1/m) * np.sum(error * stories)
    new_theta[5] = theta[5] - alpha * (1/m) * np.sum(error * mainroad)
    new_theta[6] = theta[6] - alpha * (1/m) * np.sum(error * guestroom)
    new_theta[7] = theta[7] - alpha * (1/m) * np.sum(error * basement)
    new_theta[8] = theta[8] - alpha * (1/m) * np.sum(error * hotwaterheating)
    new_theta[9] = theta[9] - alpha * (1/m) * np.sum(error * airconditioning)
    new_theta[10] = theta[10] - alpha * (1/m) * np.sum(error * parking)
    new_theta[11] = theta[11] - alpha * (1/m) * np.sum(error * prefarea)
    new_theta[12] = theta[12] - alpha * (1/m) * np.sum(error * furnishing_semi_furnished)
    new_theta[13] = theta[13] - alpha * (1/m) * np.sum(error * furnishing_unfurnished)

    return new_theta

# Initialize
theta = np.zeros(14)
alpha = 0.01
iterations = 1000
m_train = len(price_train)
cost_history = []

# Gradient Descent Loop
for i in range(iterations):
    predictions = h_of_x(theta, area_train_scaled, bedrooms_train_scaled, bathrooms_train_scaled, stories_train_scaled, mainroad_train, guestroom_train, basement_train, hotwaterheating_train, airconditioning_train, parking_train_scaled, prefarea_train, furnishing_semi_train, furnishing_unfurnished_train)
    current_cost = cost(m_train, predictions, price_train_scaled)
    cost_history.append(current_cost)
    theta = gradient_descent(theta, area_train_scaled, bedrooms_train_scaled, bathrooms_train_scaled, stories_train_scaled, mainroad_train, guestroom_train, basement_train, hotwaterheating_train, airconditioning_train, parking_train_scaled, prefarea_train, furnishing_semi_train, furnishing_unfurnished_train, alpha, predictions, price_train_scaled, m_train)
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {current_cost}")

print(f"\nFinal theta values: {theta}")
print(f"Final cost: {cost_history[-1]}")

# Evaluate
train_predictions_scaled = h_of_x(theta, area_train_scaled, bedrooms_train_scaled, bathrooms_train_scaled, stories_train_scaled, mainroad_train, guestroom_train, basement_train, hotwaterheating_train, airconditioning_train, parking_train_scaled, prefarea_train, furnishing_semi_train, furnishing_unfurnished_train)
train_predictions = (train_predictions_scaled * (price_max - price_min)) + price_min
ss_res_train = np.sum((price_train - train_predictions)**2)
ss_tot_train = np.sum((price_train - np.mean(price_train))**2)
r2_train = 1 - (ss_res_train / ss_tot_train)

test_predictions_scaled = h_of_x(theta, area_test_scaled, bedrooms_test_scaled, bathrooms_test_scaled, stories_test_scaled, mainroad_test, guestroom_test, basement_test, hotwaterheating_test, airconditioning_test, parking_test_scaled, prefarea_test, furnishing_semi_test, furnishing_unfurnished_test)
test_predictions = (test_predictions_scaled * (price_max - price_min)) + price_min
ss_res_test = np.sum((price_test - test_predictions)**2)
ss_tot_test = np.sum((price_test - np.mean(price_test))**2)
r2_test = 1 - (ss_res_test / ss_tot_test)

print(f"\nTraining R² Score: {r2_train:.4f}")
print(f"Test R² Score: {r2_test:.4f}")

# Function to predict house price
def predict_house_price(new_area, new_bedrooms, new_bathrooms, new_stories, new_mainroad,
                        new_guestroom, new_basement, new_hotwaterheating, new_airconditioning,
                        new_parking, new_prefarea, new_furnishing_semi, new_furnishing_unfurnished):

    # Scale the new data using training statistics
    new_area_scaled = (new_area - area_min) / (area_max - area_min)
    new_bedrooms_scaled = (new_bedrooms - bedrooms_min) / (bedrooms_max - bedrooms_min)
    new_bathrooms_scaled = (new_bathrooms - bathrooms_min) / (bathrooms_max - bathrooms_min)
    new_stories_scaled = (new_stories - stories_min) / (stories_max - stories_min)
    new_parking_scaled = (new_parking - parking_min) / (parking_max - parking_min)

    # Make prediction (scaled)
    predicted_price_scaled = h_of_x(theta, new_area_scaled, new_bedrooms_scaled,
                                     new_bathrooms_scaled, new_stories_scaled,
                                     new_mainroad, new_guestroom, new_basement,
                                     new_hotwaterheating, new_airconditioning,
                                     new_parking_scaled, new_prefarea,
                                     new_furnishing_semi, new_furnishing_unfurnished)

    # Unscale to get actual price
    predicted_price = (predicted_price_scaled * (price_max - price_min)) + price_min

    return predicted_price

# Example 1: Predict a house manually
print("="*60)
print("MANUAL PREDICTION EXAMPLE")
print("="*60)

predicted = predict_house_price(
    new_area=5000,              # 5000 sq ft
    new_bedrooms=3,             # 3 bedrooms
    new_bathrooms=2,            # 2 bathrooms
    new_stories=2,              # 2 stories
    new_mainroad=1,             # Yes (1) or No (0)
    new_guestroom=1,            # Yes (1) or No (0)
    new_basement=0,             # Yes (1) or No (0)
    new_hotwaterheating=0,      # Yes (1) or No (0)
    new_airconditioning=1,      # Yes (1) or No (0)
    new_parking=2,              # 2 parking spaces
    new_prefarea=1,             # Yes (1) or No (0)
    new_furnishing_semi=0,      # Semi-furnished (1) or not (0)
    new_furnishing_unfurnished=0  # Unfurnished (1) or not (0)
)

print(f"Predicted Price: ${predicted:,.2f}")