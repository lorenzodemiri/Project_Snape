# importing libraries
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading imgs
snape = Image.open("./snape.png")
library = Image.open('./library.jpg')

# loading dataframe
df = pd.read_csv('./Books.csv')
df = df.drop(columns=['Unnamed: 0'])
df.rename(columns={'1st Pub':'pub_year', 'N pag': 'N_pag'}, inplace=True)
df['Awards'].fillna('0', inplace=True)
df['Awards'] = df['Awards'].astype('int')

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

# setting header
st.title("Analysis of the 20th Century best books")
st.image(library)

# presentation of work and data
st.header('The way we developed the project :hammer_and_wrench:')
st.write("""We visited the website [GoodReads](https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century), from where we extracted a list of the best 1000 books of the 20th century as our source of information,
            by creating a program to iterate through the first 10 pages, each containing a 100 books. The program opened each book once and scraped all the required information, after which the data was saved to a dataframe in the form of a .csv file.""")
if st.button('Click here to see the DataFrame'):
        st.dataframe(df)
st.write("""The scrapped data of the books included the title, author, publication year and number of pages, the amount of ratings, reviews and awards, the average rating of the books.
        We also investigated whether the books belonged to a series or not, as well as what are the main 3 genres the book is categorized in.""")
st.write("""We created two new columns in our dataframe, for a normalization of the rating and for a normalization of the mean rating, scaling them from 0 to 10 to have a better understanding on how the rating distributes
        among each book.""")
st.write('The code we used for that is:')
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
(df['mean_norm_ratings'] - mmin) / (mmax - mmin) *10)""", language='python')

# explorating dataframe
st.header('Exploring the datas :mag:')
st.write("Here you can do some explorative work on the DataFrame by showing only the column you're interested in.")
columns_to_show = st.multiselect("Select the columns you want to display", df.columns)
st.dataframe(df[columns_to_show])

st.write("We can furthermore explore the datas. Below there are some filtering option.")
with st.beta_expander('Filter the datas by minum and maximum rating normalized from 0 to 10'):
        min_rating = st.number_input("Minimum Rating (from 0 to 10)", min_value=0, max_value=10)
        max_rating = st.number_input("Maximum Rating (from 0 to 10)", min_value=0, max_value=10, value=10)
with st.beta_expander("Or by minimum and maximum year of publication"):
        min_year = st.number_input("Minimum Year (from 1900 to 2003)", min_value=1900)
        max_year = st.number_input("Maximum Year (from 1900 to 2003)", min_value=1900, max_value=2003, value=2003)
with st.beta_expander("Or by minimum and maximum number of pages"):
        min_pag = st.number_input("Minimum Pages (from 23 to 14777)", min_value=23)
        max_pag = st.number_input("Maximum Pages (from 23 to 14777)", min_value=0, max_value=14777, value=14777)
with st.beta_expander("Or by minimum and maximum number of awards received"):
        min_award = st.number_input("Minimum Awards (from 0 to 28)", min_value=0)
        max_award = st.number_input("Maximum Awards (from 0 to 28)", min_value=0, max_value=28, value=28)
st.dataframe(df.query("@min_rating<=minmax_norm_ratings<=@max_rating & @min_year<=pub_year<=@max_year & @min_pag<=N_pag<=@max_pag & @min_award<=Awards<=@max_award"))
st.write('We can search by author')
author_input = st.text_input('Author Name', 'Author Name')
st.dataframe(df[df['Author'].str.contains(author_input)])
st.write('Or by title')
title_input = st.text_input('Title', 'Title')
st.dataframe(df[df['Title'].str.contains(title_input)])

# visualization
st.header('Visualizing the data :bar_chart:')
