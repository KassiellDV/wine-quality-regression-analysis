import numpy as np
import matplotlib.pyplot as plt
import time
import sqlite3
from typing import Tuple

def create_database(data_train: np.ndarray, data_test: np.ndarray) -> None:
    """
    Create SQLite database and store wine data
    """
    # Column names for wine data
    columns = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
        'alcohol', 'quality'
    ]

    # Create column definitions for SQL
    column_defs = ', '.join([f'{col} REAL' for col in columns])

    # Connect to database
    with sqlite3.connect('wine_quality.db') as conn:
        cursor = conn.cursor()

        # Create tables
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS training_data
                         ({column_defs})''')
        cursor.execute(f'''CREATE TABLE IF NOT EXISTS testing_data
                         ({column_defs})''')

        # Clear existing data
        cursor.execute('DELETE FROM training_data')
        cursor.execute('DELETE FROM testing_data')

        # Insert data
        placeholders = ','.join(['?' for _ in columns])
        cursor.executemany(f'INSERT INTO training_data VALUES ({placeholders})',
                         data_train)
        cursor.executemany(f'INSERT INTO testing_data VALUES ({placeholders})',
                         data_test)

        conn.commit()

def load_data_from_db() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wine data from SQLite database
    Returns:
        training_data, testing_data
    """
    with sqlite3.connect('wine_quality.db') as conn:
        cursor = conn.cursor()

        # Fetch data from tables
        cursor.execute('SELECT * FROM training_data')
        training_data = np.array(cursor.fetchall())

        cursor.execute('SELECT * FROM testing_data')
        testing_data = np.array(cursor.fetchall())

        return training_data, testing_data

def center_data(data: np.ndarray, feature_means: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center the data by subtracting means
    Args:
        data: Input data matrix
        feature_means: Means to use for centering (calculated if None)
    Returns:
        centered_data, feature_means
    """
    if feature_means is None:
        feature_means = np.mean(data, axis=0)
    centered_data = data - feature_means
    return centered_data, feature_means

def solve_normal_equations(features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve regression using normal equations method
    Returns:
        coefficients, computation_time
    """
    start_time = time.time()

    # Calculate X^T * X and X^T * y
    XtX = features.T @ features
    Xty = features.T @ target

    # Solve the system
    coefficients = np.linalg.solve(XtX, Xty)

    computation_time = time.time() - start_time
    return coefficients, computation_time

def solve_qr_decomposition(features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve regression using QR decomposition method
    Returns:
        coefficients, computation_time
    """
    start_time = time.time()

    # Compute QR decomposition
    Q, R = np.linalg.qr(features)

    # Solve the system
    Qty = Q.T @ target
    coefficients = np.linalg.solve(R, Qty)

    computation_time = time.time() - start_time
    return coefficients, computation_time

def plot_predictions(actual: np.ndarray, pred_ne: np.ndarray, pred_qr: np.ndarray) -> None:
    """
    Create comparison plot of predictions vs actual values
    """
    plt.figure(figsize=(10, 8))

    # Plot predictions
    plt.scatter(actual, pred_ne, color='blue',
               label='Normal Equations', marker='o', facecolors='none')
    plt.scatter(actual, pred_qr, color='red',
               label='QR Decomposition', marker='*')

    # Add reference line
    plt.plot([-3, 3], [-3, 3], 'k--', linewidth=1)

    # Style the plot
    plt.grid(True)
    plt.xlabel('Actual Wine Quality')
    plt.ylabel('Predicted Wine Quality')
    plt.title('Wine Quality Prediction Comparison')
    plt.legend()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.savefig('prediction_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load raw data
    print("Loading data files...")
    training_raw = np.loadtxt('dataW.txt')
    testing_raw = np.loadtxt('BdataW.txt')

    # Store in database
    print("Creating database...")
    create_database(training_raw, testing_raw)

    # Load from database
    print("Loading from database...")
    training_data, testing_data = load_data_from_db()

    # Get data dimensions
    num_train_samples, num_features = training_data.shape
    num_test_samples = testing_data.shape[0]
    print(f'Training samples: {num_train_samples}')
    print(f'Features: {num_features}')
    print(f'Testing samples: {num_test_samples}')

    # Center the training data
    training_centered, feature_means = center_data(training_data)

    # Center testing data using training means
    testing_centered, _ = center_data(testing_data, feature_means)

    # Separate features and target
    X_train = training_centered[:, :-1]  # all columns except last
    y_train = training_centered[:, -1]   # last column
    X_test = testing_centered[:, :-1]
    y_test = testing_centered[:, -1]

    # Solve using Normal Equations
    print('\nSolving using Normal Equations:')
    coefficients_ne, time_ne = solve_normal_equations(X_train, y_train)
    print(f'Time: {time_ne:.6f} seconds')

    # Solve using QR Decomposition
    print('\nSolving using QR Decomposition:')
    coefficients_qr, time_qr = solve_qr_decomposition(X_train, y_train)
    print(f'Time: {time_qr:.6f} seconds')

    # Compare solutions
    solution_difference = np.max(np.abs(coefficients_ne - coefficients_qr))
    print(f'\nDifference between solutions: {solution_difference:.10f}')

    # Make predictions
    predictions_ne = X_test @ coefficients_ne
    predictions_qr = X_test @ coefficients_qr

    # Plot results
    plot_predictions(y_test, predictions_ne, predictions_qr)

if __name__ == "__main__":
    main()
