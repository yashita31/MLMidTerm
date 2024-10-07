import pandas as pd

df = pd.read_csv("data/modified_completed_dataset10k_EleutherAI_gpt-neo-125M.csv")
df2 = pd.read_csv("data/modified_completed_dataset10k_facebook_opt-125m.csv")
df3 = pd.read_csv("data/modified_completed_dataset10k_gpt2.csv")

df_new1 = df1[:4000]
df_new2 = df2[:4000]
df_new3 = df3[:4000]

df_new1.to_csv('data/gpt-neo-4k.csv', index = False)
df_new2.to_csv('data/fb-opt-4k.csv', index = False)
df_new3.to_csv('data/gpt2-4k.csv', index = False)
df['combined_text'] = df.iloc[:, 0] + " " + df.iloc[:, 1]
print(df['combined_text'].tolist())