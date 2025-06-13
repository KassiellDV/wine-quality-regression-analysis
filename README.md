# Wine Quality Regression Analysis

## Project Overview
This project implements multiple linear regression to predict wine quality using two different solving methods: Normal Equations and QR Decomposition. The implementation demonstrates both mathematical understanding and practical software development skills with Python and SQLite database integration.

## Mathematical Background

### Data Centering
Before performing regression, we center the data by subtracting the mean:
```
x_centered = x - mean(x)
```
This process:
- Removes the mean from each feature
- Improves numerical stability
- Makes the intercept term zero

### Solving Methods
1. **Normal Equations**
   ```
   (X^T * X)β = X^T * y
   ```
   - Direct solution method
   - Computationally efficient for small datasets
   - May be numerically unstable for ill-conditioned matrices

2. **QR Decomposition**
   ```
   X = QR
   Rβ = Q^T * y
   ```
   - More numerically stable
   - Preferred for ill-conditioned matrices
   - Slightly slower but more robust

## Implementation Details

The implementation features:
- NumPy for efficient matrix operations
- SQLite database for data management
- Matplotlib for visualization
- Clean, documented code structure
- Type hints and comprehensive docstrings

## Data Structure
The wine dataset includes the following features:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## Results
Both methods (Normal Equations and QR Decomposition) produce identical results for this dataset, demonstrating:
- Numerical stability of both approaches
- Successful data centering
- Equivalent mathematical solutions

The visualization shows:
- Actual vs predicted wine quality
- Comparison between both solving methods
- Reference line for perfect predictions

## Code Structure
```
wine_quality_regression/
├── center_data.py     # Main implementation with regression methods
├── wine_quality.py    # Database handling and data preparation
├── dataW.txt         # Training data
├── BdataW.txt        # Testing data
└── README.md         # Project documentation
```

## Usage

1. Install dependencies:
   ```bash
   pip install numpy matplotlib sqlite3
   ```

2. Run the analysis:
   ```bash
   python center_data.py
   ```

## Key Features
- Data centering and preparation
- Multiple linear regression implementation
- Two solving methods comparison
- SQLite database integration
- Result visualization
- Comprehensive error handling

## Performance
- Normal Equations: ~0.0005 seconds
- QR Decomposition: ~0.0009 seconds
- Solution difference: ~0.0000000000

## Future Improvements
- Add cross-validation
- Implement feature selection
- Add more regression metrics
- Include confidence intervals
- Add regularization options

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- SQLite3
