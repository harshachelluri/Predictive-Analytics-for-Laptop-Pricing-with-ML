## **Predictive Analytics for Laptop Pricing with Machine Learning**

## **Project Overview**

SmartTech Co. partnered with our data science team to develop a reliable machine learning model for accurate laptop price prediction. The objective was to create a robust tool that could assist in strategic pricing, understand market positioning, and assess brand influence within the competitive laptop market.

### **Objectives**

- **Accurate Pricing**: Build a model capable of predicting laptop prices based on key specifications, brand reputation, and market trends.
- **Market Positioning**: Analyze how different features contribute to laptop pricing.
- **Brand Influence**: Investigate the role of brand reputation in influencing pricing.

### **Project Phases**

1. **Data Exploration and Understanding**
   - Analyzed laptop pricing trends and identified key features impacting prices.
   - Visualized feature distributions to gain initial insights into feature relationships.

2. **Data Preprocessing**
   - Handled missing values through statistical methods (e.g., mode for categorical variables, median for numerical data).
   - Detected and addressed outliers using boxplots and summary statistics.
   - Encoded categorical variables using one-hot encoding to make the data model-friendly.

3. **Feature Engineering**
   - Extracted essential features such as processor type, RAM, storage capacity, and brand.
   - Created new features that helped enhance the model's performance.

4. **Model Development**
   - **Algorithms Used**:
     - Linear Regression
     - Random Forest Regressor
     - Gradient Boosting Regressor
     - XGBoost Regressor
   - **Evaluation Metrics**:
     - Mean Absolute Error (MAE)
     - R-squared (R²)

5. **Hyperparameter Tuning**
   - Optimized the top-performing model (Random Forest) using GridSearchCV to enhance its accuracy.

6. **Real-time Prediction**
   - Developed a mechanism for predicting prices of newly released laptops with unseen configurations.

7. **Interpretability**
   - Identified the key features with the highest impact on pricing, including processor type, brand reputation, RAM, and storage.


### **Tools and Technologies**

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Model Evaluation**: GridSearchCV for hyperparameter tuning

### **Results**

- **Top Performing Model**: Random Forest Regressor
  - R²: 0.872
  - MAE: 0.164

- **Key Insights**:
  - Processor type, RAM, storage, and brand reputation were found to be the most significant factors influencing pricing.
  - Budget laptops showed lower prediction errors compared to high-end models, highlighting the model’s better accuracy for lower-priced products.

### **Deploy**
- Deploy the model via a web interface to provide real-time price predictions for new laptops, ensuring SmartTech Co. stays competitive in the market by making data-driven pricing decisions.

