from linear_regression import LinearRegression, load_data
# import pandas as pd
# from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    
    model = LinearRegression()
    x, y = load_data('data.csv')
    model.train(x, y)
    print(model.predict(6000))
    print(model.precision(x, y))