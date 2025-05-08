# CMSC320-FINAL

1. Header with contributions
  Title
  Spring 2025 Data Science Project
  Group Members: Ayush Joshi, Andy Sun, Ashwin Subbu, Adarsa Pedada, Kristopher Stoichkov, Richard Suwanto
  Contributions:


2. Introduction
  Source of Data: https://www.kaggle.com/datasets/hugomathien/soccer
  For our final project, we looked at player statistics from professional soccer leagues. It contains informatoin on over 25,000 matches and 11,000 players. It has all sorts of player attributes like strength, speed, aggression, etc. These attributes were obtained from EA Sports' FIFA games. 
We wanted to observe relationships between different player attributes using different statistical tests. 

3. Data Curation
In this step, we collect data. Let's load the SQLite database.  
```
import sqlite3
import pandas as pd
import csv
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```
Here are the libraries we need. We also load the data. Lets start by opening a connecting to our database. The cursor object lets use SQL commands. 

```
conn = sqlite3.connect('database.sqlite')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

for table_name in tables:
    table = table_name[0]
    cursor.execute(f"SELECT * FROM {table}")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    with open(f"{table}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(rows)

    print(f"Exported {table} to {table}.csv")

conn.close()
```
Then we loop through the tables and lists them out. Each table is saved as a seperate csv. 
Now lets clean the data! 
Specifically, we narrow down on two tables: Player.csv and Player_Attributes.csv.

```
player_df = pd.read_csv("Player.csv")
player_attributes_df = pd.read_csv("Player_Attributes.csv")

player_df['birthday'] = pd.to_datetime(player_df['birthday'])
player_attributes_df['date'] = pd.to_datetime(player_attributes_df['date'])
latest_dates = player_attributes_df.groupby('player_api_id')['date'].max().reset_index()
player_df = player_df.merge(latest_dates, on='player_api_id', how='left')
player_df['age'] = (((player_df['date'] - player_df['birthday']).dt.days)/365.25).astype(int)
```

First we need to load the CSVs into dataframes. Then we can convert the birthdays into a datetime format, making it easier to work with. 
Why? Well now we can easily calculate the age of a player. We create a new column called age that shows how old a player is. 

We also need to fill in any potential missing data points. Here we group by players and use forward and backward fill to fill gaps. 
```
grouped_players = player_attributes_df.groupby('player_api_id')[['sprint_speed', 'acceleration']]
filled = grouped_players.transform(lambda x: x.ffill().bfill())
player_attributes_df[['sprint_speed', 'acceleration']] = filled[['sprint_speed', 'acceleration']]
```

Now we merge the attributes into player data, grabbing the most recent sprint speed and acceleration values for eac player. We then add them to the dataframe. 
```
most_recent_attributes = pd.merge(
    latest_dates,
    player_attributes_df,
    on=['player_api_id', 'date'],
    how='left'
)
most_recent_attributes = most_recent_attributes[['player_api_id', 'acceleration', 'sprint_speed']]
player_df = player_df.merge(most_recent_attributes,on='player_api_id', how='left')
player_df
```

Perfect! We are now ready to analyze our data. 

4. Exploratory Data Analysis
  Now lets actually explore our data and try to find patterns or correlations between the data. 
Lets see if age affects spring speed and acceleration in players. Does getting older cause your agility to suffer? 
Real quick, lets go over null hypotheses and hypothesis testing.
The null hypothesis states that there is no relationship between the data groups. So here, we would say that the null hypothesis is that age does not affect spring speed and acceleration.
The alternative hypotehsis states that there is a relationship. The alternative hypothesis here is that there is a relationship between age and spring speed/acceleration. 
After performing a statistical test, we get a p-value. This gives the probability of observing your data point if the null hypothesis is true. 
So if the p-value is above a significant value (i.e. 0.05) we would reject the null hypothesis. 
We will use a two-sample t-test to compare 2 age groups which are split by median age. 
```
median_age = player_df['age'].median()
younger_sprint = player_df[player_df['age'] <= median_age]['sprint_speed']
older_sprint = player_df[player_df['age'] > median_age]['sprint_speed']
younger_acceleration = player_df[player_df['age'] < median_age]['acceleration']
older_acceleration = player_df[player_df['age'] >= median_age]['acceleration']
t_stat_sprint, p_val_sprint = stats.ttest_ind(younger_sprint, older_sprint, equal_var=False)
print(f"T-test comparing sprint speed by median age split:")
print(f"t = {t_stat_sprint}, p = {p_val_sprint}")
t_stat_accel, p_val_accel = stats.ttest_ind(younger_acceleration, older_acceleration, equal_var=False)
print(f"T-test comparing acceleration by median age split:")
print(f"t = {t_stat_accel}, p = {p_val_accel}")
```
We filtered the spring speed and acceleration for both groups. 
After that, we use a t-test to check if the mean spring speed differs between age groups. EXPLAIN T_TEST AND RESULTS FURTHER.
We also do the same test for acceleration. 

```
sns.boxplot(x='age', y='sprint_speed', data=player_df)
plt.title('Sprint Speed by Age Group')
plt.show()

sns.boxplot(x='age', y='acceleration', data=player_df)
plt.title('Acceleration by Age Group')
plt.show()
```
Now we can plot boxplots with the seaborn library. 

Now lets see if there is any correlation between a player's vision and aggression. To give some background, higher vision means that players are able to "see" the field better. Essentially players with high vision can make better passes. 
The aggression stat determines how "aggressive" you are on the field. PLayers with a high aggression stat are stronger and tend to make aggresive plays on both offense and defense. However, they are also more likely to get penalties. 

```
grouped_players3 = player_attributes_df.groupby('player_api_id')[['vision', 'aggression']]
filled3 = grouped_players3.transform(lambda x: x.ffill().bfill())
player_attributes_df[['vision', 'aggression']] = filled3[['vision', 'aggression']]

plot1 = plt.scatter(player_attributes_df["vision"], player_attributes_df["aggression"])
spearman = stats.spearmanr(player_attributes_df["vision"], player_attributes_df["aggression"])
spearman_pvalue = spearman.pvalue
if spearman_pvalue > 0.05:
    print("Higher than alpha")
else:
    print("Lower than alpha")

pvalue = 3.26e-80
display(f"pvalue is {pvalue}")

plt.xlabel("Vision")
plt.ylabel("Aggression")
plt.title("Correlationship Between Vision and Aggression")
plt.show()
```
We used the spearman test.
HA: Players vision does have an effect on their aggressiveness.
When looking at the p-value of the Spearman graph, we end up with a number lower than the alpha (0.05). Therefore, we reject the null hypothesis.

When looking at the graph and the r-value correlation, you can see there is a slight negative correlation between the player's aggression and vision. The correlation exists, but it is not strong. However, to some extent, players with lower vision may tend to be more aggressive than those who have better vision.


For our last part, we compared strength and defensive work rate. 
```
grouped_players_2 = player_attributes_df.groupby('player_api_id')[['defensive_work_rate', 'strength']]
filled_2 = grouped_players_2.transform(lambda x: x.ffill().bfill())
player_attributes_df[['defensive_work_rate', 'strength']] = filled_2[['defensive_work_rate', 'strength']]


most_recent_workrates = pd.merge(
    latest_dates,
    player_attributes_df,
    on=['player_api_id', 'date'],
    how='left'
)
most_recent_workrates = most_recent_workrates[['player_api_id', 'defensive_work_rate', 'strength']]
player_df = player_df.merge(most_recent_workrates,on='player_api_id', how='left')
player_df
```
Since there are 3 groups (high, medium, and low) we used the anova test. This is a test that specifically analyzes the difference between means of more than two groups. 

5. Primary Analysis

6. Visualization

7. Insights and Conclusion 
