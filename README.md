# Prime-Nest-Housing-Prediction

# House Price Prediction using Linear Regression

## Overview
This project is a **House Price Prediction Model** built using **Linear Regression** in Python. The goal is to develop a predictive model that estimates house prices based on various features such as location, size, number of bedrooms, and other influencing factors. The model achieved an accuracy of **1.00**, indicating a perfect fit on the dataset.

## Dataset
The dataset used for this project contains various attributes affecting house prices, including:
- **House Size (sq ft)**
- **Number of Bedrooms**
- **Number of Bathrooms**
- **Lot Size**
- **Neighborhood Rating**
- **Year Built**
- **Proximity to Schools and Hospitals**
- **Price** (Target Variable)


### Data Source
The dataset was sourced from **writing a python code to create syntethic data to carry on this project**, and was preprocessed before training the model.

## Technologies Used
- **Python**
- **Pandas** (Data Manipulation)
- **NumPy** (Numerical Computation)
- **Matplotlib & Seaborn** (Data Visualization)
- **Scikit-Learn** (Model Training & Evaluation)

## Data Preprocessing
Before training the model, the dataset was cleaned and prepared:
1. **Handling Missing Values**: Missing data was imputed using the mean/mode where necessary.
2. **Encoding Categorical Variables**: Converted categorical features (like neighborhood rating) into numerical values.
3. **Feature Scaling**: Standardized numerical features to ensure better model performance.
4. **Outlier Removal**: Detected and removed extreme values that could affect model accuracy.

## Model Building
The model was built using **Linear Regression** due to its simplicity and effectiveness for this type of predictive analysis.

### Steps Taken:
1. **Split the Data**: The dataset was split into training (80%) and testing (20%) sets.
2. **Feature Selection**: Selected the most relevant features for model training.
3. **Training the Model**: Used Scikit-learn's `LinearRegression` model to fit the training data.
4. **Model Evaluation**: Measured model performance using metrics like:
   - **R-squared Score (R²)**
   - **Mean Absolute Error (MAE)**
   - **Mean Squared Error (MSE)**
   - **Root Mean Squared Error (RMSE)**

## Model Performance
- **R² Score**: 1.00 (Indicating a perfect model fit)
- **Mean Absolute Error (MAE)**: 0.0
- **Mean Squared Error (MSE)**: 0.0
- **Root Mean Squared Error (RMSE)**: 0.0

The model achieved **100% accuracy**, meaning it perfectly predicts house prices on the given dataset. However, such a high score could indicate potential overfitting, requiring further validation on unseen data.

## Results & Insights
- The most influential factors in house pricing included **house size**, **number of bedrooms**, and **location**.
- The model successfully captured linear relationships between features and price.
- A perfect accuracy score suggests the dataset might be too small or lacks variance, which may require further testing with real-world data.

## How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script:
   ```sh
   jupyter notebook House_Price_Prediction.ipynb
   ```
   or
   ```sh
   python house_price_prediction.py
   ```

## Future Improvements
- **Collect and Use a Larger Dataset** to validate the model on real-world data.
- **Apply Regularization Techniques** (Ridge or Lasso Regression) to avoid overfitting.
- **Try Advanced Models** like Random Forest, XGBoost, or Neural Networks for better performance.
- **Deploy the Model** using Flask or FastAPI to make it accessible via a web app.

## Conclusion
This project successfully demonstrates how **Linear Regression** can be used to predict house prices with high accuracy. While achieving an R² score of **1.00** is ideal, further testing on different datasets is recommended to ensure model robustness.

## Contact
For any queries or collaborations, feel free to connect:
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/olu-samuel-89246b22a/)

---

