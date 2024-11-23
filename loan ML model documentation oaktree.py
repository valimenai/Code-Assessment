"""The focus of the machine learning model for predicting loan defaults was to first establish a clear understanding of the data we were working with, and how each column correlated to the loan_status. We assume in this exercise that loan_status 0 meant in good standing, and that loan_status 1 meant the loan was troubled or defaulted. 

The first step was to ensure the dataframe was clean. Then, we analyze the columns to establish a correlation matrix. After getting a better understanding of the dataframe, we can begin working on the model. 
The focus of this model is to predict the loan_status. In order to do that, we first had to rank each column’s correlation with respect to loan_status. The findings were that previous_loan_defaults_on_file had the strongest correlation to loan_status, followed by loan_to_income_ratio, loan_percent_income, loan_int_rate, person_home_ownership, and person_income. Other important factors are borrower_risk_score, credit_score, regional_unemployment_rate, and loan_intent. These are the columns we focus on, with the assumption being that these will allow for the most robust predictions. We create  custom columns,  debt_burden, income_per_dependent, risk_bucket, and credit_income_ratio, to better train the model (specifics of the calculations can be found in the code repository). The decision to focus on the aforementioned columns solely is due to the fact that the other columns had very low correlations with loan_status, and so can create superfluous noise in a model. 

The findings were that these correlations are a strong predictor of loan_status in the model. It maintained high accuracy for loan default prediction (92%), with a higher degree of precision (68%), mitigating false positives and allowing business leaders to make better informed decisions without unnecessarily diverting resources to loans that would not in reality default. 

While the model is performing reasonably well at an overall of accuracy of 89%, there are myriad factors that can still be taken into account to improve this model. Access to census data that would inform demographics, such as zip code and job occupation, are two important factors (to help predict income stability). Other factors include estimated net worth (liquid and illiquid), current other debt outstanding, so on and so forth. With access to a wider set of data, this model could be trained to even higher accuracy. 

"""