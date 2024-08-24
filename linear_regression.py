'''
1) Predict the price of a car for a given mileage
2) Train the model using the data in data.csv
3) Bonus: plotting the data into a graph to see their repartition
4) Bonus: plotting the line resulting into the same graph
5) Bonus: calculate the precision of the model
'''


class LinearRegression:
    def __init__(self, theta0=0, theta1=0, learning_rate=0.001, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta0 = theta0
        self.theta1 = theta1
        self.x = []
        self.y = []
    
    def mean(self, x):
        return sum(x) / len(x)
    
    def std(self, x):
        mean_x = self.mean(x)
        return (sum([(xi - mean_x) ** 2 for xi in x]) / len(x)) ** 0.5

    # Normalization using mean and standart deviation
    def normalize(self, xi):
        if len(self.x) == 0:
            return xi
        m = len(self.x)
        mean_x = self.mean(self.x)
        std_x = self.std(self.x)
        return (xi - mean_x) / std_x
    
    def denormalize_theta(self, theta0, theta1):
        theta0 = theta0 - (theta1 * self.mean(self.x)) / self.std(self.x)
        theta1 = theta1 / self.std(self.x)
        return theta0, theta1

    
    # Predict for normalized x
    def predict_norm(self, x):
        return self.theta0 + self.theta1 * x
    
    # # Predict for unnormalized x
    def predict(self, new_x):
        new_x = self.normalize(new_x)
        return self.predict_norm(new_x)

    def train(self, x, y):
        self.x = x
        self.y = y
        x_norm = [self.normalize(xi) for xi in x]
        m = len(x)

        for _ in range(self.epochs):

            for i in range(m):
                prediction = self.predict_norm(x_norm[i])
                error = prediction - y[i]

                theta_tmp0 = self.learning_rate * sum([error]) / m
                theta_tmp1 = self.learning_rate * sum([error * x_norm[i]]) / m

                self.theta0 -= theta_tmp0
                self.theta1 -= theta_tmp1

        return self.denormalize_theta(self.theta0, self.theta1)
    
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
    
