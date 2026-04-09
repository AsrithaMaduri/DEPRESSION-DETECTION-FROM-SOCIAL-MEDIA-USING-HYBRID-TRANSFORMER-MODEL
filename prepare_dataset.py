import pandas as pd

# load original dataset
df = pd.read_csv("data/Mental-Health-Twitter.csv")

print("Original columns:")
print(df.columns)

# keep only needed columns
df = df[["post_text", "label"]]

# rename column
df = df.rename(columns={"post_text": "text"})

# remove missing values
df = df.dropna()

# shuffle dataset
df = df.sample(frac=1, random_state=42)

# save new dataset
df.to_csv("data/dataset.csv", index=False)

print("Dataset prepared successfully!")
print("New dataset shape:", df.shape)