
from sklearn.model_selection import train_test_split
import torch
from kan import KAN
import copy


class KAN_Classifier:
    def __init__(self,grid=3, k=3,steps=50) -> None:
        self.dataset = {}
        self.model = None
        self.model_list = []

        self.grid = grid
        self.k = k
        self.steps = steps
        
    
    def fit(self,X,y):
        y = y.astype(int)

        if self.model is None:

            model = KAN(width=[X.shape[1],X.shape[1]*2+1,2], grid=self.grid, k=self.k)
            self.model = model

        model = copy.deepcopy(self.model)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        self.dataset['train_input'] = torch.from_numpy(X_train)
        self.dataset['test_input'] = torch.from_numpy(X_test)
        self.dataset['train_label'] = torch.from_numpy(y_train)
        self.dataset['test_label'] = torch.from_numpy(y_test)
    
        try:
            model.train(self.dataset, opt="LBFGS", steps=self.steps,loss_fn=torch.nn.CrossEntropyLoss())
        except:
            model = self.model_list[-1]
        self.model_list.append(model)    

    def predict(self,X):
        model = self.model_list[-1]
        return torch.argmax(model(torch.from_numpy(X)),dim=1).detach().numpy()

    
class KAN_Regressor:
    def __init__(self,grid=3, k=3,steps=50) -> None:
        self.dataset = {}
        self.model = None
        self.model_list = []

        self.grid = grid
        self.k = k
        self.steps = steps

    def fit(self,X,y):
        if self.model is None:
            model = KAN(width=[X.shape[1],X.shape[1]*2+1,1], grid=self.grid, k=self.k)
            self.model = model

        model = copy.deepcopy(self.model)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        self.dataset['train_input'] = torch.from_numpy(X_train)
        self.dataset['test_input'] = torch.from_numpy(X_test)
        self.dataset['train_label'] = torch.from_numpy(y_train[:,None])
        self.dataset['test_label'] = torch.from_numpy(y_test[:,None])

        try:
            model.train(self.dataset, opt="LBFGS", steps=self.steps)
        except:
            model = self.model_list[-1]

        self.model_list.append(model)    

    def predict(self,X):
        model = self.model_list[-1]
        return model(torch.from_numpy(X)).detach().numpy()
        
    
def ackley_function(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) \
           - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) \
           + np.e + 20




def show_cla():
    # Generate random data points
    x = np.random.uniform(-5, 5, 100)
    y = np.random.uniform(-5, 5, 100)
    
    # Calculate z values using the Ackley function
    z = ackley_function(x, y)
    
    # Combine x and y into a feature matrix
    XY = np.vstack([x, y]).T

    # Calculate the median z value
    median_z = np.percentile(z, 50)
    # Assign labels based on whether z is less than or equal to the median
    labels = z <= median_z
    labels = labels.astype(int) 

    # Create and fit a KAN classifier model
    model = KAN_Classifier()
    model.fit(XY, labels)

    # Create a grid for evaluation
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    z_grid = ackley_function(x_grid, y_grid)
    
    # Create a feature matrix for the grid
    XY_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    # Predict labels for the grid
    labels_pred = model.predict(XY_grid).reshape(x_grid.shape)

    # Create subplots for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the actual classes
    CS1 = ax1.contourf(x_grid, y_grid, z_grid, cmap='viridis')
    scatter = ax1.scatter(x, y, c=labels, cmap='RdYlBu', edgecolor='k', alpha=0.7)
    ax1.set_title('Actual Classes')
    fig.colorbar(CS1, ax=ax1)
    fig.colorbar(scatter, ax=ax1, boundaries=[-0.5, 0.5, 1.5], ticks=[0, 1])
    
    # Plot the KAN class predictions
    CS2 = ax2.contourf(x_grid, y_grid, labels_pred, cmap='viridis', levels=[-0.5, 0.5, 1.5])
    ax2.set_title('KAN Class Predictions')
    fig.colorbar(CS2, ax=ax2, boundaries=[-0.5, 0.5, 1.5], ticks=[0, 1])
    
    # Set labels for the axes
    for ax in (ax1, ax2):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # Show the plot
    plt.show()






def show_reg():
    # Generate random data points
    np.random.seed(0)
    x = np.random.uniform(-5, 5, 100)
    y = np.random.uniform(-5, 5, 100)
    z = ackley_function(x, y) + 0.2 * np.random.normal(size=x.shape)  # Add some noise

    # Combine x and y into a feature matrix
    XY = np.vstack([x, y]).T


    # degree = 5
    # model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model = KAN_Regressor()
    model.fit(XY, z)

    # Create a grid for evaluation
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    z_grid = ackley_function(x_grid, y_grid)

    # Predict points on the grid using the model
    XY_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    z_pred = model.predict(XY_grid).reshape(x_grid.shape)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
    # Original data and Ackley function
    ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, cmap='viridis')
    ax1.scatter(x, y, z, color='red')
    ax1.set_title('Original Ackley Function')
    # Fitted result
    ax2.plot_surface(x_grid, y_grid, z_pred, alpha=0.3, cmap='viridis')
    ax2.set_title('KAN Regression Fit')
    for ax in (ax1, ax2):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.show()



if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # show_reg()
    show_cla()


    