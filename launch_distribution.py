import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("../Datasets/HealthPlatforms_BEP.xlsx")
df.info
df.columns

df["LAUNCH YEAR"].value_counts(sort=False).plot(kind="line", figsize=(6, 7))
plt.xlabel("LAUNCH YEAR")
plt.ylabel("count")
plt.title("Distribution of Launched Companies per Year")
plt.show()
