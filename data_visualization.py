import pandas as pd
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn
from distfit import distfit
import re
from scipy import stats
#"Link","Title","Author","Rating Count","Review Count","Rating Value","N pag","1st Pub","series","Genres","Awards"
df = pd.read_csv("./resources/Books.csv")
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
def get_plothist_bestrated(df):
    df_sort_ratings = df.sort_values(by='minmax_norm_ratings', ascending=False)
    df_rating_sliced = df_sort_ratings[:10]
    plt.figure(figsize=(10, 10))
    plt.barh(df_rating_sliced['Title'], df_rating_sliced['minmax_norm_ratings'])
    plt.ylabel('Book Title')
    plt.xlabel('Rating from 1 to 10')
    plt.title('Best 10 Rated Books')
    plt.show()
def get_plthist_bestawarded(df):
    df_sort_award = df.sort_values(by='Awards', ascending=False)
    df_award_sliced = df_sort_award[:10]
    plt.figure(figsize=(10,10))
    plt.barh(df_award_sliced['Title'], df_award_sliced['Awards'])
    plt.ylabel('Book Title')
    plt.xlabel('Number of awards')
    plt.title('Best 10 Books by number of Awards')
    plt.show()
##KINBERLEY VISUALIZATION
def get_plot_minmax_norm_distr(df):
    df = df.sample(n = 100)
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["minmax_norm_ratings"], label='MinMax Ratings Norm', bins=20)
    c=plt.legend()
    plt.xlabel('Ratings', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Density', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('Min-Max Norm Ratings Distribution', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.savefig('minmax_norm.jpg')
    plt.show()
def get_plot_mean_norm_distr(df):
    df = df.sample(n = 100)
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["mean_norm_ratings"], label='Mean Norm', color='red', bins=20)
    c=plt.legend()
    plt.xticks(range(1, 10))
    plt.xlabel('Ratings', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Density', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('Mean Norm Ratings Distribution', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.show()
def get_plt_minmax_mean_norm_distr(df):
    df = df.sample(n = 100)
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["minmax_norm_ratings"], label='MinMax Norm', color='blue')
    seaborn.distplot(df["mean_norm_ratings"], label='Mean Norm', color='red')
    c=plt.legend()
    c=plt.legend()
    plt.xticks(range(1, 10))
    plt.xlabel('Ratings', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Density', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('Comparison of Norm Ratings Distributions', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
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
def get_plt_minmax_meand_norm_distr(df):
    df = df.sample(n = 100)
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["minmax_norm_ratings"], label='MinMax Awards Norm', color='blue')
    seaborn.distplot(df["mean_norm_ratings"], label='Mean Awards Norm', color='red')
    c=plt.legend()
    plt.xticks(range(1, 10))
    plt.xlabel('Awards', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Density', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('Comparison of Norm Awards Distributions', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.show()
def get_plt_prob_awards(df):
    mean_awards = df.Awards.mean()
    mean_ratings = df.Ratings.mean()
    population_variance_awards = df.Awards.var(ddof=0)
    sample_variance_awards = df.Awards.var()
    population_variance_ratings = df.Ratings.var(ddof=0)
    sample_variance_ratings = df.Ratings.var() 
    population_std_awards = df.Awards.std(ddof=0)
    sample_std_awards = df.Awards.std()
    population_std_ratings = df.Ratings.std(ddof=0)
    sample_std_ratings = df.Ratings.std()

    plt.figure(figsize=(15,10))
    x_awards = 3.0
    seaborn.distplot(df["Awards"], label='Awards', kde=False, fit=stats.norm, color='green')
    plt.text(x_awards-0.55, .01, '$x$='+str(x_awards), rotation='vertical', fontsize=15, fontweight='bold', color='black')
    xplot_awards = np.linspace(min(df["Awards"]), x_awards, 10)
    yplot_awards = stats.norm.pdf(xplot_awards, mean_awards, population_std_awards)
    plt.fill_between(xplot_awards,  yplot_awards)
    c=plt.legend(fontsize=15)
    plt.xticks(range(-5, 29))
    plt.vlines(x=mean_awards, ymin=0, ymax=0.8, colors='darkgreen', linestyles='dashed', label='mean')
    plt.vlines(x=mean_awards+population_std_awards, ymin=0, ymax=0.8, colors='black', linestyles='dotted', label='SD')
    plt.vlines(x=mean_awards-population_std_awards, ymin=0, ymax=0.8, colors='black', linestyles='dotted', label='SD')
    plt.xlabel('Awards', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Density', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('PDF of Three Awards or Less', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.show()
def get_plt_prob_rating(df):
    p_ratings_new = 1 - 0.22
    z_ratings_new = stats.norm.ppf(p_ratings_new)
    mean_ratings = df.Ratings.mean()
    population_std_ratings = df.Ratings.std(ddof=0)
    x_ratings_new = z_ratings_new * population_std_ratings + mean_ratings
    plt.figure(figsize=(15,10))
    seaborn.distplot(df["Ratings"], label='Rating', kde=False, fit=stats.norm, color='green')
    plt.text(x_ratings_new-0.04, .1, '$x$='+str(x_ratings_new), rotation='vertical', fontsize=15, fontweight='bold', color='black')

    xplot_ratings_new = np.linspace(x_ratings_new, max(df["Ratings"]), 50)
    yplot_ratings_new = stats.norm.pdf(xplot_ratings_new, mean_ratings, population_std_ratings)
    plt.fill_between(xplot_ratings_new,  yplot_ratings_new)
    c=plt.legend(fontsize=15)
    plt.xticks([3,3.5,4,4.5,5])
    plt.vlines(x=mean_ratings, ymin=0, ymax=2.2, colors='darkgreen', linestyles='dashed', label='mean')
    plt.vlines(x=mean_ratings+population_std_ratings, ymin=0, ymax=2.2, colors='black', linestyles='dotted', label='SD')
    plt.vlines(x=mean_ratings-population_std_ratings, ymin=0, ymax=2.2, colors='black', linestyles='dotted', label='SD')
    plt.xlabel('Rating', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Density', labelpad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.title('PDF of 4.21 Rating or More', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.show()
def get_plt_awards_ratings_norm_distr(df):
    plt.figure(figsize=(15,10))
    ax = seaborn.distplot(df["Awards"], label='Awards')
    ax = seaborn.distplot(df["Ratings"], label='Ratings')
    c=plt.legend(fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('Grey')
    ax.spines['bottom'].set_color('Grey')
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    ax.set_xlabel('Ratings/Awards', labelpad=10, color='darkblue', fontsize=14)
    ax.set_ylabel('Density', labelpad=10, color='darkblue', fontsize=14)
    ax.set_title('Awards/Ratings Normal Distributions', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 18, 'fontweight': 'bold'})
    plt.show()
def get_boxplot_distr(df):
    plt.figure(figsize=(15,10))
    ax = seaborn.boxplot(data=[df["Awards"], df["Ratings"]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('Grey')
    ax.spines['bottom'].set_color('Grey')
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    plt.xticks(np.arange(2), ["Awards", "Ratings"], fontsize=14, color='darkblue')
    ax.set_ylabel('Density', labelpad=10, color='darkblue', fontsize=14)
    ax.set_title("Boxplot of Awards/Ratings Normal Distributions", pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 18, 'fontweight': 'bold'})
    plt.show()
def get_plt_distr_awards_ratings_norm_distr(df):
    zscores_awards_array = stats.mstats.zscore(df["Awards"])
    zscores_ratings_array = stats.mstats.zscore(df["Ratings"])
    plt.figure(figsize=(15,10))
    ax = seaborn.distplot(zscores_awards_array, label='Awards')
    ax = seaborn.distplot(zscores_ratings_array, label='Ratings')
    c=plt.legend(title='Normalized values (Z-Scores)', fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('Grey')
    ax.spines['bottom'].set_color('Grey')
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    ax.set_xlabel('Ratings/Awards', labelpad=10, color='darkblue', fontsize=14)
    ax.set_ylabel('Density', labelpad=10, color='darkblue', fontsize=14)
    ax.set_title('Distribution Plot of Normalised Distributions', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 18, 'fontweight': 'bold'})
    plt.show()
def get_plt_norml_distr(df):
    plt.figure(figsize=(15,10))
    zscores_awards_array = stats.mstats.zscore(df["Awards"])
    zscores_ratings_array = stats.mstats.zscore(df["Ratings"])
    ax = seaborn.boxplot(data=[zscores_awards_array, zscores_ratings_array])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('Grey')
    ax.spines['bottom'].set_color('Grey')
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    plt.xticks(np.arange(2), ["Awards", "Ratings"], fontsize=14, color='darkblue')
    ax.set_ylabel('Density', labelpad=10, color='darkblue', fontsize=14)
    ax.set_title("Boxplot of Normalised Distributions", pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 18, 'fontweight': 'bold'})
    plt.show()
def get_hist_awards_genres(df):
    genres = df["Genres"].tolist()
    genres = [eval(x) for x in genres]
    genres_list = [re.sub("[\[\]',\"\n\s]", "", item) for sublist in genres for item in sublist]
    title_repeated = df.Title.repeat(3).tolist()
    year_repeated = df['1st Pub'].repeat(3).tolist()
    awards_repeated = df.Awards.repeat(3).tolist()
    ratings_repeated = df["Rating Value"].repeat(3).tolist()
    big_df = pd.DataFrame({'Title': title_repeated ,'Genres': genres_list, 'Year': year_repeated, 'Awards': awards_repeated, 'Rating': ratings_repeated})
    big_df['Awards'] = big_df['Awards'].fillna(0)
    genres_df = big_df.groupby("Genres")["Awards"].sum()
    genres_df = genres_df.reset_index().sort_values("Awards", axis=0, ascending=False)
    genres_df = genres_df.loc[genres_df["Awards"]>100]
    fig, ax = plt.subplots()
    bars = ax.bar(x=genres_df["Genres"], height=genres_df["Awards"])
    plt.xticks(rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('Grey')
    ax.spines['bottom'].set_color('Grey')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )
    ax.set_xlabel('Genre', labelpad=10, color=bar_color)
    ax.set_ylabel('Number of Awards', labelpad=10, color=bar_color)
    ax.set_title('Total Number of Awards by Top Genres', pad=25, color='DarkBlue', weight='bold', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    fig.tight_layout()
    plt.show()
def get_hist_mean_awards_genres(df):
    genres = df["Genres"].tolist()
    genres = [eval(x) for x in genres]
    genres_list = [re.sub("[\[\]',\"\n\s]", "", item) for sublist in genres for item in sublist]
    title_repeated = df.Title.repeat(3).tolist()
    big_df = pd.DataFrame({'Title': title_repeated ,'Genres': genres_list, 'Year': year_repeated, 'Awards': awards_repeated, 'Rating': ratings_repeated})
    big_df['Awards'] = big_df['Awards'].fillna(0)
    genre_means = big_df.groupby("Genres").mean()
    genre_means["Total"] = big_df.groupby("Genres").size()
    genre_means = genre_means.reset_index().sort_values("Total", axis=0, ascending=False)
    genre_means["Year"] = genre_means['Year'].fillna(method='pad').astype(int)
    best_genres = genre_means.loc[genre_means["Total"]>70]
    fig, ax = plt.subplots()
    bars = ax.bar(x=best_genres["Genres"], height=best_genres["Awards"])
    plt.xticks(rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('Grey')
    ax.spines['bottom'].set_color('Grey')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )
    ax.set_xlabel('Genre', labelpad=10, color=bar_color)
    ax.set_ylabel('Mean Awards', labelpad=10, color=bar_color)
    ax.set_title('Mean Number of Awards by Top Genres', pad=25, color='DarkBlue', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    fig.tight_layout()
    fig.tight_layout()
    plt.show()
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
def get_plotscatter_genres_occurence(df):
    genre = df["Genres"]
    genretypes = genre.values.tolist()
    genretypes = [eval(x) for x in genretypes]
    for x in genretypes:
        if len(x)!=3:
            x.append('Fantasy')
    flat_list = [re.sub("[\[\]',\"\n\s]", "", item) for sublist in genretypes for item in sublist]
    genreocc = dict()
    for key in flat_list:
        if key in genreocc:
            genreocc[key] +=1
        else:
            genreocc[key] =1
    T_genre = list(zip(* [iter(flat_list)]*3))
    T_genre = [tuple(x) for x in T_genre]
    df['Genres'] = T_genre
    dgenres = df.groupby("Genres").agg({"Genres": np.size})
    genreamount = pd.DataFrame(T_genre)
    genretypeamount = pd.DataFrame(flat_list)
    genretypeamount.columns = ["GenreType"]
    dg = genretypeamount.groupby("GenreType").agg({"GenreType": [lambda x: np.size(x)]})
    dg.columns = ["Amount of books in genre"]
    dg['Genres'] = dg.index
    gt = dg["Amount of books in genre"].tolist()
    gg = dg["Genres"].tolist()
    plt.figure(figsize = (15,20))
    plt.scatter(gt, gg, label = "Amount")
    plt.xlabel('Amount')
    plt.ylabel('Genre types')
    plt.legend(loc='lower right')
    plt.title('Genre types occurence')
    plt.grid(True, linewidth= 1, linestyle="--")
    plt.show()
    
if __name__ == "__main__":
    get_hist_awards_genres(df)
    get_comparation_awarded_series_book(df)
    #get_plotline_table_awards_books(df)
    #plot_distr_minmax_norm_distr(df)
    #get_proportion_book_awards(df)
    #get_comparation_awarded_series_book(df)
    #get_count_series_book(df)
    #get_count_awarded_book(df)
    #plot_mean_norm_distr(df)
    #get_fitted_model(df, data_select = "mean_norm_ratings")

