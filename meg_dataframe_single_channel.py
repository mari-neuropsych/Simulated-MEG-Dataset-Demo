import pandas as pd

# Convert the first channel to a DataFrame
df = pd.DataFrame({
    "Time (s)": times,
    "MEG0": data[0]  # First channel
})

# Show the first 5 rows
print(df.head())
