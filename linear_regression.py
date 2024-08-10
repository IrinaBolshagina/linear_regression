# 1) Predict the price of a car for a given mileage
# 2) Train the model using the data in data.csv
# 3) Bonus: plotting the data into a graph to see their repartition
# 4) Bonus: plotting the line resulting into the same graph
# 5) Bonus: calculates the precision


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

# Normalize data 
def normalize_data(x):
    return [(x[i] - min(x)) / (max(x) - min(x)) for i in range(len(x))]



class LinearRegression:

    def __init__(self, theta0=0, theta1=0, learning_rate=0.000000001, epochs=100):
        self.theta0 = theta0
        self.theta1 = theta1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.x = []
        self.y = []
        self.x_norm = []


    # Normalize one value of x between 0 and 1
    def normalize(self, new_x):
        return (new_x - min(self.x)) / (max(self.x) - min(self.x))

    
    # y = theta0 + theta1 * x
    def predict(self, mileage):
        if len(self.x) == 0:
            raise ValueError('You must train the model before making predictions')
        mileage = self.normalize(mileage)
        return self.theta0 + (self.theta1 * mileage)
    

    # Gradient Descent
    def train(self, x, y):
        self.x = x
        self.y = y
        self.theta0 = 0
        self.theta1 = 0
        sum_theta0 = 0
        sum_theta1 = 0
        m = len(x)
        for i in range(self.epochs):
            for j in range(m):
                sum_theta0 += self.predict(x[j]) - y[j]
                sum_theta1 += (self.predict(x[j]) - y[j]) * self.normalize(x[j])
            self.theta0 -= (self.learning_rate * sum_theta0) / m
            self.theta1 -= (self.learning_rate * sum_theta1) / m
        return self.theta0, self.theta1
    

    def gradient_descent(self, x, y):
        self.theta0 = 0
        self.theta1 = 0
        sum_theta0 = 0
        sum_theta1 = 0
        m = len(x)
        for i in range(self.epochs):
            predictions = self.predict(x)
            sum_theta0 = sum([predictions[i] - y[i] for i in range(m)])
            sum_theta1 = sum([(predictions[i] - y[i]) * x[i] for i in range(m)])
            self.theta0 -= (self.learning_rate * sum_theta0) / m
            self.theta1 -= (self.learning_rate * sum_theta1) / m
        return self.theta0, self.theta1
    

    # Mean Squared Error
    def precision(self, x, y):
        sum_error = sum([(self.predict(x[i]) - y[i]) ** 2 for i in range(len(x))])
        return sum_error / len(x)


    def plot_data(self):
        x = [self.data[i][0] for i in range(len(self.data))]
        y = [self.data[i][1] for i in range(len(self.data))]
        plt.plot(x, y, 'ro')
        plt.axis([0, max(x), 0, max(y)])
        plt.show()


    def plot_line(self):
        x = [self.data[i][0] for i in range(len(self.data))]
        y = [self.data[i][1] for i in range(len(self.data))]
        plt.plot(x, y, 'ro')
        plt.axis([0, max(x), 0, max(y)])
        plt.plot([0, max(x)], [self.predict(0), self.predict(max(x))], 'b')
        plt.show()
