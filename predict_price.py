from linear_regression import LinearRegression
from train import load_data
import sys

def load_thetas(file):

    try:
        with open(file, 'r') as file:
            lines = file.readlines()
            return float(lines[0].split(',')[0]), float(lines[0].split(',')[1])
    except:
        return 0,0

def predict(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


if __name__ == '__main__':

    # Read result of training - if file doesn't exist, define thetas 0,0
    try:
        thetas = load_thetas('thetas.csv')
    except: 
        thetas = (0,0)
    
    # Get mileage from user
    mileage = input("\nEnter the mileage of the car in km: ")
    try:
        mileage = int(mileage)
    except:
        exit("\nMileage must be a positive integer\n")
    
    # Predict the price of the car
    price = round(predict(mileage, thetas[0], thetas[1]), 2)
    if price < 0:
        print("\nEstimated price of the car is 0. The mileage is too high.\n")
    else:
        print(f"\nThe estimated price of the car is {price} euros\n")
