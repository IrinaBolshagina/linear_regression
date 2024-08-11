'''
1) Predict the price of a car for a given mileage
2) Train the model using the data in data.csv
3) Bonus: plotting the data into a graph to see their repartition
4) Bonus: plotting the line resulting into the same graph
5) Bonus: calculate the precision of the model
'''


# Read scv file and return 2 lists of x and y
def load_data(file):
    x = []
    y = []
    try:
        with open(file, 'r') as file:
            lines = file.readlines()
            for i in range(1, len(lines)):
                x.append(int(lines[i].split(',')[0]))
                y.append(int(lines[i].split(',')[1]))
        return(x, y)
    except Exception as e:
        exit(e)


class LinearRegression:
    def __init__(self, theta0=0, theta1=0, learning_rate=0.001, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta0 = theta0
        self.theta1 = theta1
        self.x = []
        self.y = []

    # Normalization using mean and standart deviation
    def normalize(self, xi):
        m = len(self.x)
        mean_x = sum(self.x) / m
        std_x = (sum([(xi - mean_x) ** 2 for xi in self.x]) / m) ** 0.5
        return (xi - mean_x) / std_x
    
    # Predict for normalized x
    def predict_norm(self, x):
        return self.theta0 + self.theta1 * x
    
    # Predict for unnormalized x
    def predict(self, new_x):
        if len(self.x) == 0:
            raise ValueError("Model has not been trained yet.")
        new_x = self.normalize(new_x)
        return self.predict_norm(new_x)

    def train(self, x, y):
        self.x = x
        self.y = y
        x_norm = [self.normalize(xi) for xi in x]
        m = len(x)

        for _ in range(self.epochs):
            sum_theta0 = 0
            sum_theta1 = 0

            for i in range(m):
                prediction = self.predict_norm(x_norm[i])
                error = prediction - y[i]
                sum_theta0 += error
                sum_theta1 += error * x_norm[i]

            self.theta0 -= (self.learning_rate * sum_theta0) / m
            self.theta1 -= (self.learning_rate * sum_theta1) / m

        return self.theta0, self.theta1
    
    # Sum of squared error by regression line
    def ssr(self, x, y):
        m = len(x)
        sum_error = sum([(self.predict(x[i]) - y[i]) ** 2 for i in range(m)])
        return sum_error
    
    # Sum of squared error by mean line
    def ssm(self, x, y):
        m = len(x)
        mean_y = sum(y) / m
        sum_error = sum([(y[i] - mean_y) ** 2 for i in range(m)])
        return sum_error
    
    # Mean squared error
    def mean_squared_error(self):
        return self.ssr(self.x, self.y) / len(self.x)
    
    def mean_absolute_error(self):
        return sum([abs(self.predict(self.x[i]) - self.y[i]) for i in range(len(self.x))]) / len(self.x)

    # R2 score of the model - how well the model fits the data
    def score(self):
        return 1 - self.ssr(self.x, self.y) / self.ssm(self.x, self.y)
    
