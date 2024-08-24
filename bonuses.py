from linear_regression import LinearRegression
from train import load_data
from matplotlib import pyplot as plt

def visualise_dataset(x, y):
    plt.title('Price vs Mileage')
    plt.xlabel('mileage, km')
    plt.ylabel('price, euro')
    plt.scatter(x, y)
    plt.show()

def visualise_predictions(x, y, model):
    plt.title('Price vs Mileage')
    plt.xlabel('mileage, km')
    plt.ylabel('price, euro')
    plt.scatter(x, y)
    x_points = [min(x), max(x)]
    y_points = [model.predict(x) for x in x_points]
    plt.plot(x_points, y_points, color='red')
    plt.legend(['real data', 'predicted'])
    plt.show()

if __name__ == '__main__':
    
    model = LinearRegression(0,0)

    x, y = load_data('data.csv')
    model.train(x, y)
    
    # show graphs
    visualise_dataset(x, y)
    visualise_predictions(x, y, model)

    # show metrics
    print("\nLinear regression precision:")
    print("\nR2 score:")
    print(model.score())
    print("\nMean squared error:")
    print(model.mean_squared_error())
    print("\nMean absolute error:")
    print(model.mean_absolute_error())
    print()
