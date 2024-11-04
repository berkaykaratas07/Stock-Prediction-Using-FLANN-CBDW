from reflame import MhaFlnnRegressor


def initialize_model():
    """
    Initializes the MhaFlnnRegressor model with specified parameters.

    Uses the Firefly Algorithm (FFA) for optimization and Legendre polynomial
    expansion for feature enhancement.

    Returns:
        MhaFlnnRegressor: An initialized MhaFlnnRegressor model with defined settings.
    """
    # Define optimization parameters for the model
    opt_paras = {"name": "FFA", "epoch": 25, "pop_size": 70}

    # Initialize the model with specified parameters
    model = MhaFlnnRegressor(
        expand_name="legendre",  # Legendre polynomial expansion for feature enhancement
        n_funcs=3,
        act_name="relu",
        obj_name="RMSE",
        optimizer="OriginalFFA",  # Firefly Algorithm as the optimizer
        optimizer_paras=opt_paras,
        verbose=True
    )
    return model


def train_model(model, X_train_scaled, y_train_scaled):
    """
    Trains the provided model using the scaled training data.

    Parameters:
        model (MhaFlnnRegressor): The initialized model to be trained.
        X_train_scaled (array-like): Scaled training set of independent variables.
        y_train_scaled (array-like): Scaled training set of dependent variable.

    Returns:
        MhaFlnnRegressor: The trained model.
    """
    # Fit the model to the scaled training data
    model.fit(X_train_scaled, y_train_scaled)
    return model


def evaluate_model(model, X_test_scaled, y_test_scaled):
    """
    Evaluates the model on the scaled test data using multiple metrics.

    Parameters:
        model (MhaFlnnRegressor): The trained model to be evaluated.
        X_test_scaled (array-like): Scaled test set of independent variables.
        y_test_scaled (array-like): Scaled test set of dependent variable.

    Returns:
        tuple: RMSE value and a dictionary containing additional evaluation metrics.
    """
    # Calculate RMSE (Root Mean Square Error) for model evaluation
    rmse = model.score(X=X_test_scaled, y=y_test_scaled, method="RMSE")

    # Calculate additional metrics such as R2, NSE, MAPE, and MAE
    metrics = model.scores(X=X_test_scaled, y=y_test_scaled, list_methods=["R2", "NSE", "MAPE", "MAE"])

    return rmse, metrics
