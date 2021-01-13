# importing libraries
import streamlit as st
from PIL import Image
import pandas as pd

# loading imgs
snape = Image.open("snape.png")
library = Image.open('library.jpg')

# loading dataframe
df = pd.read_csv('Books.csv')
df = df.drop(columns=['Unnamed: 0'])

# setting sidebar
st.sidebar.header('Team Snape')
st.sidebar.image(snape, width=100)
st.sidebar.markdown("""Team Snape is composed by:\n
                    Kimberley\n
                    Bence\n
                    Lorenzo\n
                    Luca\n
                    Zakariya""")
st.sidebar.markdown("The repository for the project can be found [here](https://github.com/lorenzodemiri/Project_Snape)!")

# setting headers
st.title("Analysis of the 20th Century best books")
st.image(library)
st.header('How did we developed the project')

# presentation of work and data
st.write("""We took [GoodReads](https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century) list of the best book of the 20th century as our data source,
            create a program to scrape all the information we needed from there and saved it to a dataframe.""")
st.write('To see the dataframe, please use the button below.')
if st.button("Visualize DataFrame"):
    st.dataframe(df)
st.write("""Straight from the site we got the title, author, rating count, review count, rating value, number of pages, year of the first publication,
        if the book belongs to a series or not, genres, and how many awards the book won""")
st.write("""Based on those datas, we decided to do a normalization of the rating and of the mean rating, scaling them from 0 to 10 to have a better understanding on how the rating distributes
        among each book""")
st.write("""The formula we used for that are:""")
st.code("""# getting needed variable for the calculations
max_rating = df['Rating Value'].max()
min_rating = df['Rating Value'].min()
range_of_ratings = max_rating - min_rating
mean_rating = df['Rating Value'].mean()

# calculating minmax norm ratings
round(1 + 9*((df['Rating Value'] - min_rating)/range_of_ratings) , 3)

# calculating mean norm ratings
round(1 + 9*((df['Rating Value'] - mean_rating)/range_of_ratings) , 3)

# scaling mean norm ratings from 0 to 10
mmax = np.max(df['mean_norm_ratings'])
mmin = np.min(df['mean_norm_ratings'])
(df['mean_norm_ratings'] - mmin) / (mmax - mmin) *10)""")

# explorating dataframe
st.write("""Here you can do some explorative work on the DataFrame by showing only the column you're interested in.""")
columns_to_show = st.multiselect("Select the columns you want to display", df.columns)