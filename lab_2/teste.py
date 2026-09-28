import pandas as pd

def phi_vector2(history_df, day_feature_df, p, q):

    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df = history_df.sort_values("date", ascending=False).reset_index(drop=True)
    features = {}
    df_phi = pd.DataFrame()
    column_names = history_df.columns.tolist()

    for i in range(1,p) :


            features[f'{column_names[1]}_lag_{i}'] = history_df.loc[i-1, f'{column_names[1]}']

    for name in column_names[2:]:

        for i in range(q) :

            if i == 0:
               features[f'{name}_lag_{i}'] = day_feature_df[name].iloc[0]

            else:

                features[f'{name}_lag_{i}'] = history_df.loc[i-1, f'{name}']

    df_phi = pd.DataFrame([features])
  
    return df_phi

# 1. Fake History Data (6 days total)
data_history = {
    'date': ['2023-10-06', '2023-10-01', '2023-10-04', '2023-10-02', '2023-10-05', '2023-10-03'],
    'chlorophyll': [6.6, 1.1, 4.4, 2.2, 5.5, 3.3], # Matches the dates (e.g. 1st = 1.1)
    'temperature': [60, 10, 40, 20, 50, 30],
    'target': [600, 100, 400, 200, 500, 300] 
}
df_history_test = pd.DataFrame(data_history)

# 2. Fake Day Feature Data
data_day = {
    'temperature': [10],
    'target': [100] 
}
df_day_test = pd.DataFrame(data_day, index=['target_day'])

# 3. Run the function
df_phi_result2 = phi_vector2(df_history_test, df_day_test, p=4, q=2)


print("--- Original Fake Data (Out of order) ---")
print(df_history_test)
print("\n--- Output of phi_vector (Sorted & Grabbed latest 2 days) ---")
print("\n--- Output of phi_vector2 (Sorted & Grabbed latest 2 days) ---")
print(df_phi_result2)