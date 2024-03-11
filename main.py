#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Reading EEG data
data = pd.read_csv("emotions.csv")
print(data.info())

sample = data.loc[0, 'fft_0_b':'fft_749_b']

#EEG data vizualisation
plt.figure(figsize=(16, 10))
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
plt.show()