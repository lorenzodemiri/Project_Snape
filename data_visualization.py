import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
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

def  get_plot_pag_rating(df):
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
def plot_distr_minmax_norm(df):
    import seaborn as sns
    sample_df = df.sample(n = 100)
    minmax_list = sample_df["minmax_norm_ratings"].tolist()
    minmax_list.sort()
    mean_norm = sample_df["mean_norm_ratings"].tolist()
    mean_norm.sort()
    
    sns.distplot(minmax_list, label='MinMax Norm')
    c=plt.legend()
    
    sns.distplot(mean_norm, label='Mean Norm', color='red')
    c1=plt.legend()
    
    sns.distplot(minmax_list, label='MinMax Norm', color='blue')
    sns.distplot(mean_norm, label='Mean Norm', color='red')
    c2=plt.legend()

#BENCE VISUALIZATION
def mean_norm(books):
    max_rating = books['Rating Value'].max()
    min_rating = books['Rating Value'].min()
    range_of_ratings = max_rating - min_rating

    books['minmax_norm_ratings'] = round(1 + 9*((books['Rating Value'] - min_rating)/range_of_ratings) , 3)

    mean_rating = books['Rating Value'].mean()

    books['mean_norm_ratings'] = round(1 + 9*((books['Rating Value'] - mean_rating)/range_of_ratings) , 3)
    dr = books[["Title", "1st Pub", 'minmax_norm_ratings']]
    year = dr.groupby("1st Pub").agg({"minmax_norm_ratings": [lambda x: np.mean(x)]})
    dyear.columns = ["Mean of norm ratings"]

    dyear['publishing year'] = dyear.index()

    pubyear = dyear["publishing year"].tolist()
    meannorm = dyear["Mean of norm ratings"].tolist()


    plt.figure(figsize = (15,15))
    plt.scatter(pubyear, meannorm, label = "Mean norm of the year")
    plt.xlabel('Year')
    plt.ylabel('Scale of 1-10')
    plt.legend(loc='lower right')
    plt.title('Mean norm of books based on release year')
    plt.grid(True, linewidth= 1, linestyle="--")

    plt.xticks(np.arange(1900, 2010, step=5))
    plt.show()

if __name__ == "__main__":
    #print(df)
    get_correlation_table(df)
    get_correlation_coe(df)
    #get_plot_norm_rating_award(df)
    get_coe_rating_award(df)
    get_plot_norm_rating_award(df)
    #get_plot_pag_rating(df)
    #f.to_csv('./resources/Books.csv') 
    


#df['minmax_norm_ratings'] = dx['minmax_norm_ratings']
#df['mean_norm_ratings'] = dx['mean_norm_ratings']
#print(df['minmax_norm_ratings'])
#df = count_awards(df)
#sequence = ["Title","Author","Rating Count","Review Count","Rating Value","N pag","1st Pub","series","Genres","Awards","minmax_norm_ratings","mean_norm_ratings","Link"]
#df = df.reindex(columns = sequence)
