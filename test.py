from linear_regression import LinearRegression, load_data

if __name__ == '__main__':

    model = LinearRegression()

    x, y = load_data('data.csv')
    
    # print(x,y)
    # print(model.predict([100000, 200000]))
    print(model.train(x,y))
    print(model.predict(5000))
    print(model.gradient_descent(x,y))
    # print(model.predict([100000, 200000]))
    print(model.predict(5000))
    # print(model.precision(x,y))