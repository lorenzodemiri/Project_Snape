## LUCA'S FUNCTION VISUALIZATION

- ### get_correlation_table(df)
Print and return the correlation table
- ### ge_plot_pag_rating(df)
Print the scatter plot of pages and number of ratings
- ### get_correlation_coe(df)
Print and return the correlation coefficient for number of pages and number of ratings
- ### get_plotscatter_norm_rating_award(df)
Print the scatter plot of normalized ratings and number of awards
- ### get_coe_rating_award(df):
Print and return correlation coefficient for ratings and number of awards
- ### get_plotbar_norm_rating_award(df):
Print the bar chart of normalized ratings and number of awards

## BENCE'S FUNCTION VISUALIZATION
- ### get_plotscatter_meannormbook_realeseyear(df, enable_line = True):
Print the scatter plot and the table of Mean Normalization Rating based on the publishing year, and return as well the dataframe containing the data. 
**enable_line set by default on true**
- ### get_plotpair_minmax_mean_normrating(df):
Printing the pair plot and comparing rating value, minmax normalized ratings and mean normalized ratings
- ### get_plothist_minmax_mean_normrating(df):
Printing the histogram plot and comparing rating value, minmax normalized ratings and mean normalized ratings
- ### def get_fitted_model(df, data_select = "minmax_norm_ratings")
Printing the data and the plot of fitting model for the value set on the parameter **data_select**
- ### get_prediction_model(df, data_select = "mean_norm_ratings")
Printing the data and the plot of prediction model for the value set on the parameter **data_select**
- ### get_plotline_table_awards_books(df):
Printing the plotline and table that compare given awards and books 