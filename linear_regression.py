# 1) Predict the price of a car for a given mileage
# 2) Train the model using the data in data.csv
# 3) Bonus: plotting the data into a graph to see their repartition
# 4) Bonus: plotting the line resulting into the same graph
# 5) Bonus: calculate the precision


# Read scv file and return 2 lists of x and y
def load_data(file):
    x = []
    y = []
    with open(file, 'r') as file:
        lines = file.readlines()
        for i in range(1, len(lines)):
            x.append(int(lines[i].split(',')[0]))
            y.append(int(lines[i].split(',')[1]))
    return(x, y)


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta0 = 0
        self.theta1 = 0
        self.mean = 0
        self.std = 0

    # Normalization using mean and standart deviation

    def calculate_mean(self, x):
        return sum(x) / len(x)

    def calculate_std(self, x):
        mean_x = self.calculate_mean(x)
        return (sum([(xi - mean_x) ** 2 for xi in x]) / len(x)) ** 0.5

    def normalize(self, xi):
        m = len(self.x)
        mean = sum(self.x) / m
        std = (sum([(xi - mean) ** 2 for xi in self.x]) / m) ** 0.5
        return (xi - mean) / std
    
    # Predict for normalized x
    def predict_norm(self, x):
        return self.theta0 + self.theta1 * x
    
    # Predict for unnormalized x
    def predict(self, new_x):
        new_x = self.normalize(new_x)
        return round(self.predict_norm(new_x), 2)

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


    
    # Mean absolut persentage error
    # def precision(self, x, y):
    #     m = len(x)
    #     sum_error = sum([abs(self.predict(x[i]) - y[i]) / y[i] for i in range(m)])
    #     return (sum_error / m) * 100
    
    def precision(self, x, y):
    # Check if y has any zero values to avoid division by zero errors
        if any(val == 0 for val in y):
            raise ValueError("Target values contain zeros, which may cause division by zero.")

        m = len(x)
        # Compute the MAPE
        sum_error = sum([abs(self.predict(x[i]) - y[i]) / max(y[i], 1e-10) for i in range(m)])
        return (sum_error / m) * 100
