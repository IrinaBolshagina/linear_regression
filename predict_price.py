from linear_regression import LinearRegression, load_data

if __name__ == '__main__':
    
    model = LinearRegression(0,0)

    x, y = load_data('data.csv')
    model.train(x, y)
    
    mileage = input("\nEnter the mileage of the car in km: ")
    try:
        mileage = int(mileage)
    except:
        exit("\nMileage must be a positive integer\n")
    
    price = round(model.predict(mileage), 2)
    if price < 0:
        print("\nEstimated price of the car is 0. The mileage is too high.\n")
    else:
        print(f"\nThe estimated price of the car is {price} euros\n")