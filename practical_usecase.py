'''
Note: Run logistic_regression_all_var.py first before running this script,
as this file depends on the model, imputer, and scaler defined there.
'''

year_number = 5  #Number should be corresponding to the year of the dataset
years_remaining = 6 - year_number

file_path = "data/Test_firm.xlsx"

sheets = ["Healthy Firm", "Distressed Firm"]

for sheet in sheets:
    new_firm = pd.read_excel(file_path, sheet_name=sheet)
    new_firm = new_firm[X.columns]
    new_firm = new_firm.apply(pd.to_numeric, errors="coerce")
    new_firm = imputer.transform(new_firm)
    new_firm_scaled = scaler.transform(new_firm)
    prob = model.predict_proba(new_firm_scaled)[0][1]
    print(sheet)
    print(
        f"From the training data, firms with similar financial ratios went bankrupt about {round(prob*100,2)}% of the time within the next {years_remaining} year(s)."
    )
    if prob >= 0.3:
        print("High Bankruptcy Risk")	
    else:
        print("Low Bankruptcy Risk")