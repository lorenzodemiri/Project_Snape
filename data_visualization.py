import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn
from distfit import distfit
#"Link","Title","Author","Rating Count","Review Count","Rating Value","N pag","1st Pub","series","Genres","Awards"
df = pd.read_csv("./Project_Snape/resources/Books.csv")
#dx = pd.read_csv("./books.csv")
#print(df['Awards'])

def count_awards_func(string_award):
    if string_award is not np.NaN:
        string_award = string_award.split(",")
        return len(string_award)
    else:
        return np.NaN

def count_awards(df):
    df['Awards'] = df['Awards'].apply(count_awards)
    return df

def mix_max_norm_rating(books):
    max_rating = books['Rating Value'].max()
    min_rating = books['Rating Value'].min()
    range_of_ratings = max_rating - min_rating
    books['minmax_norm_ratings'] = round(1 + 9*((books['Rating Value'] - min_rating)/range_of_ratings) , 3)
    mean_rating = books['Rating Value'].mean()
    books['mean_norm_ratings'] = round(1 + 9*((books['Rating Value'] - mean_rating)/range_of_ratings) , 3)
    dr = books[["Title", "1st Pub", 'minmax_norm_ratings']]
    return dr

def dyear(df):
    dyear = df.groupby("1st Pub").agg({"minmax_norm_ratings": [lambda x: np.mean(x)]})
    dyear.columns = ["Mean of norm ratings"]
    dyear['publishing year'] = dyear.index
    dyear = dyear.style.hide_index()

##LUCA VISUALIZATION
def get_correlation_table(df):
    correlation_table = df.corr()
    display(correlation_table)
    return correlation_table
def get_plot_pag_rating(df):
    df_sampled = df.sample(n=100)
    # creating scatter plot of pages and number of ratings
    plt.figure(figsize=(10,10))
    slope, intercept = np.polyfit(df_sampled['N pag'], df_sampled['Rating Count'], 1)
    plt.plot(df_sampled['N pag'], slope * df_sampled['N pag'] + intercept, color='black') # code for the regression line
    plt.ylim(2)
    plt.scatter(df_sampled['N pag'], df_sampled['Rating Count'])
    plt.xlabel('Number of pages')
    plt.ylabel('Number of ratings')
    plt.title('Number of pages and number of ratings')
    plt.show()
    # exporting plot
    #plt.savefig('pages_ratings.jpg')
    return
def get_correlation_coe(df):
    correlation_coefficient = df['N pag'].corr(df['Rating Count'])
    print("Correlation coefficient for number of pages and number of rating is: ", correlation_coefficient)
    return correlation_coefficient
def get_plotscatter_norm_rating_award(df):
    df_dropped = df[['minmax_norm_ratings', 'Awards']]
    df_droppped = df_dropped.dropna(inplace=True)
    plt.figure(figsize=(10,10))
    slope, intercept = np.polyfit(df_dropped['minmax_norm_ratings'], df_dropped['Awards'], 1)
    plt.plot(df_dropped['minmax_norm_ratings'], slope*df_dropped['minmax_norm_ratings'] + intercept, color='black')
    plt.scatter(df_dropped['minmax_norm_ratings'], df_dropped['Awards'])
    plt.xlabel('Ratings (normalized)')
    plt.ylabel('Number of awards')
    plt.title('Ratings and number of awards')
    #plt.savefig('ratings_awards.jpg')
    plt.show()
    return
def get_coe_rating_award(df):
    cc_npages_nrating = df['minmax_norm_ratings'].corr(df['Awards'])
    print("Correlation coefficient for ratings and number of awards is: ", cc_npages_nrating)
    return cc_npages_nrating
def get_plotbar_norm_rating_award(df):
    df_sampled = df.sample(n=100)
    plt.figure(figsize=(10, 10))
    plt.bar(df_sampled['minmax_norm_ratings'], df_sampled['Awards'])
    plt.xlabel('Ratings (normalized)')
    plt.ylabel('Number of awards')
    plt.title('Ratings and number of awards')
    plt.show()

##KINBERLEY VISUALIZATION
def get_plot_minmax_norm_distr(df):
    df = df.sample(n = 100)
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["minmax_norm_ratings"], label='MinMax Norm', bins=20)
    c=plt.legend()
    plt.title("Min-Max Norm Distribution")
    plt.savefig('minmax_norm.jpg')
    plt.show()
def get_plot_mean_norm_distr(df):
    df = df.sample(n = 100)
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["mean_norm_ratings"], label='Mean Norm', color='red', bins=20)
    c=plt.legend()
    plt.title("Mean Norm Distribution")
    plt.xticks(range(1, 10))
    plt.savefig('mean_norm.jpg')
    plt.show()
def get_plt_minmax_mean_norm_distr(df):
    df = df.sample(n = 100)
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["minmax_norm_ratings"], label='MinMax Norm', color='blue')
    seaborn.distplot(df["mean_norm_ratings"], label='Mean Norm', color='red')
    c=plt.legend()
    plt.title("Comparison of Norm Distributions")
    plt.show()
def get_count_series_book(df):
    df["series"] = df.series.replace({0: "No", 1: "Yes"})  
    series = df.groupby(['series']).count()
    series = series.rename(columns = {"Unnamed: 0":"Series Sum"})
    display(series["Series Sum"])
    return series["Series Sum"]
def get_count_awarded_book(df):
    df['Awards'] = df['Awards'].fillna(0)
    df['Awards'].values[df['Awards'].values > 0] = 1
    df["Awards"] = df.Awards.replace({0.0: "No", 1.0: "Yes"})  
    awards = df.groupby(['Awards']).count()
    awards = awards.rename(columns = {"Unnamed: 0":"Awards Sum"})
    display(awards["Awards Sum"])
    return awards["Awards Sum"]
def get_proportion_awards_book(df):
    df_res = get_count_awarded_book(df)
    print("Proportion of books with one or more awards:")
    prob_award = df_res['Yes']/ (df_res['Yes'] + df_res['No'])
    display(prob_award)
def get_comparation_awarded_series_book(df):
    df["series"] = df.series.replace({0: "No", 1: "Yes"})  
    df["Awards"] = df.Awards.replace({0.0: "No", 1.0: "Yes"})  
    series_award = df.groupby("Awards")["series"].agg([lambda z: np.sum(z=="Yes"), "size"])
    series_award.columns = ["Also With Series", "With Awards?"]
    display(series_award)
    return series_award

#BENCE VISUALIZATION
def get_plotscatter_meannormbook_realeseyear(df, enable_line = True):
    range_of_ratings = df['Rating Value'].max() - df['Rating Value'].min()
    df['minmax_norm_ratings'] = round(1 + 9*((df['Rating Value'] - df['Rating Value'].min())/range_of_ratings) , 3)
    df['mean_norm_ratings'] = round(1 + 9*((df['Rating Value'] - df['Rating Value'].mean())/range_of_ratings) , 3)  
    dr = df[["Title", "1st Pub", 'minmax_norm_ratings']]
    dyear = dr.groupby("1st Pub").agg({"minmax_norm_ratings": [lambda x: np.mean(x)]})
    dyear.columns = ["Mean of norm ratings"]
    dyear['publishing year'] = dyear.index
    display(dyear)
    plt.figure(figsize = (15,15))
    plt.scatter(dyear["publishing year"], dyear["Mean of norm ratings"], label = "Mean norm of the year")
    if enable_line: plt.plot(dyear["publishing year"], dyear["Mean of norm ratings"], color='red')
    plt.xlabel('Year')
    plt.ylabel('Scale of 1-10')
    plt.legend(loc='lower right')
    plt.title('Mean norm of books based on release year')
    plt.grid(True, linewidth= 1, linestyle="--")
    plt.xticks(np.arange(1900, 2010, step=5))
    #plt.savefig('nameoftheplot.jpg')
    plt.show()
    return dyear
def get_plotpair_minmax_mean_normrating(df):
    seaborn.pairplot(df, vars=('Rating Value', 'minmax_norm_ratings', 'mean_norm_ratings'), kind='reg')
    plt.show()
def get_plothist_minmax_mean_normrating(df):
    plt.figure(figsize = (15,15))
    plt.hist(df["Rating Value"])
    plt.hist(df["minmax_norm_ratings"])
    plt.hist(df["mean_norm_ratings"])
    plt.show()
def get_fitted_model(df, data_select = "minmax_norm_ratings"):
    y = [0,1,2,3,4,5,6,7,8,9,10]
    dist = distfit(alpha=0.05, smooth=10)
    dist.fit_transform(df[data_select])
    best_distr = dist.model
    display(best_distr)
    dist.summary
    dist.plot_summary()
    plt.show()
def get_make_prediction(df, data_select = "mean_norm_ratings"):
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dist = distfit(alpha=0.05, smooth=10)
    dist.fit_transform(df[data_select])
    best_distr = dist.model
    dist.summary
    dist.predict(y)
    dist.y_pred
    dist.y_proba
    np.array([0.02040816, 0.02040816, 0.02040816, 0.        , 0.        ])
    dist.plot()
    plt.show()
def get_plotline_table_awards_books(df):
    data_aw = df.groupby('Awards')['Awards'].count()
    display(data_aw)
    plt.figure(figsize = (15,15))
    plt.xticks(np.arange(0, 30, step=1))
    plt.yticks(np.arange(0, 300, step=10))
    plt.xlabel('Amount of awards')
    plt.ylabel('Amount of books')
    plt.plot(data_aw)
    plt.show()
    

if __name__ == "__main__":
    #get_plotline_table_awards_books(df)
    #plot_distr_minmax_norm_distr(df)
    #get_proportion_book_awards(df)
    #get_comparation_awarded_series_book(df)
    #get_count_series_book(df)
    #get_count_awarded_book(df)
    #plot_mean_norm_distr(df)
    #get_fitted_model(df, data_select = "mean_norm_ratings")
 
