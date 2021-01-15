# importing libraries
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading imgs
snape = Image.open("./webapp/snape.png")
library = Image.open('./webapp/library.jpg')
best_award_book = Image.open('./webapp/best_award_books.jpg')
best_rated_book = Image.open('./webapp/best_rated_books.jpg')
pages_rating = Image.open('./webapp/pages_ratings.jpg')
norm_rating_dists = Image.open('./webapp/norm_ratings_dists.jpg')
best_fit = Image.open('./webapp/MeanNormRatings_BestFit.jpg')
book_awards = Image.open('./webapp/bookawards.jpg')
ratings_awards = Image.open('./webapp/ratings_awards.jpg')
mean_awards_genres = Image.open('./webapp/MeanAwardsByGenre.jpg')
total_awards_genres = Image.open('./webapp/BarChartGenres_Awards.jpg')
python_image = Image.open('./webapp/python_image.jpg')
pandas_image = Image.open('./webapp/pandas_image.jpg')
streamlit_image = Image.open('./webapp/streamlit_image.jpg')
matplotlib_image = Image.open('./webapp/matplotlib_image.jpg')
numpy_image = Image.open('./webapp/numpy_image.jpg')
seaborn_image = Image.open('./webapp/seaborn_image.jpg')

# loading dataframe
df = pd.read_csv('./webapp/Books.csv')
df = df.drop(columns=['Unnamed: 0'])
df.rename(columns={'1st Pub':'pub_year', 'N pag': 'N_pag'}, inplace=True)
df['Awards'].fillna('0', inplace=True)
df['Awards'] = df['Awards'].astype('int')


# setting sidebar
st.sidebar.header('Team Snape')
st.sidebar.image(snape, width=100)
st.sidebar.markdown("Team Snape is composed by:")
st.sidebar.markdown("""> [![KimGit](https://img.shields.io/badge/Kimberley-Git-blue&?style=plastic&logo=github&?labelColor=grey&?logoWidth=200&?)](https://github.com/T-A-Y-L-O-R-S-T-R-I-V-E)
>
> [![BenceGit](https://img.shields.io/badge/Bence-Git-blue&?style=plastic&logo=github&?labelColor=grey&?logoWidth=200&?)](https://github.com/kovacsbelsen)
>
> [![LucaGit](https://img.shields.io/badge/Luca-Git-blue&?style=plastic&logo=github&?labelColor=grey&?logoWidth=200&?)](https://github.com/lpianta)
>
> [![ZakariyaGit](https://img.shields.io/badge/Zakariya-Git-blue&?style=plastic&logo=github&?labelColor=grey&?logoWidth=200&?)](https://github.com/ZakariyaM27)
>
> [![LorenzoGit](https://img.shields.io/badge/Lorenzo-Git-blue&?style=plastic&logo=github&?labelColor=grey&?logoWidth=100&?)](https://github.com/lorenzodemiri)
""")

st.sidebar.markdown("The repository for the project can be found [here](https://github.com/lorenzodemiri/Project_Snape)!")

# setting header
st.title("Analysis of the 20th Century best books")
st.image(library)

# presentation of work and data
st.header('The project :books:')
st.write("""The amount of books published in the 20th century is massive. To navigate better in this sea of paper we need to collect data and ask them some questions.
        We visited the website [GoodReads](https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century), from where we extracted a list of the best books of the 20th century as our source of data,
        by creating a program to scrape all the information we needed from the pages and we saved it to a dataframe.""")
if st.button('Click here to see the DataFrame'):
        st.dataframe(df)



# explorating dataframe
st.header('Exploring the data :mag:')
st.write("Here you can do some exploration of the DataFrame by showing only the columns you're interested in.")
columns_to_show = st.multiselect("Select the columns you want to display", df.columns)
st.dataframe(df[columns_to_show])

st.write("We can furthermore explore the data. See some filtering options below.")
with st.beta_expander('Filter by minimum and maximum rating normalized from 0 to 10'):
        min_rating = st.number_input("Minimum Rating (from 0 to 10)", min_value=0, max_value=10)
        max_rating = st.number_input("Maximum Rating (from 0 to 10)", min_value=0, max_value=10, value=10)
with st.beta_expander("Filter by minimum and maximum year of publication"):
        min_year = st.number_input("Minimum Year (from 1900 to 2003)", min_value=1900)
        max_year = st.number_input("Maximum Year (from 1900 to 2003)", min_value=1900, max_value=2003, value=2003)
with st.beta_expander("Filter by minimum and maximum number of pages"):
        min_pag = st.number_input("Minimum Pages (from 23 to 14777)", min_value=23)
        max_pag = st.number_input("Maximum Pages (from 23 to 14777)", min_value=0, max_value=14777, value=14777)
with st.beta_expander("Filter by minimum and maximum number of awards received"):
        min_award = st.number_input("Minimum Awards (from 0 to 28)", min_value=0)
        max_award = st.number_input("Maximum Awards (from 0 to 28)", min_value=0, max_value=28, value=28)
st.dataframe(df.query("@min_rating<=minmax_norm_ratings<=@max_rating & @min_year<=pub_year<=@max_year & @min_pag<=N_pag<=@max_pag & @min_award<=Awards<=@max_award"))
with st.beta_expander('Search by author'):
        author_input = st.text_input('Author Name', 'Author Name')
        st.dataframe(df[df['Author'].str.contains(author_input)])
with st.beta_expander('Search by title'):
        title_input = st.text_input('Title', 'Title')
        st.dataframe(df[df['Title'].str.contains(title_input)])

# visualization
st.header('Visualizing the data :bar_chart:')
st.write("There are a lot of questions we can ask about the best books of the 20th Century. We tried to answer some of them")
with st.beta_expander('Which are the best rated books?'):
        st.image(best_rated_book)
with st.beta_expander('Which are the books that won more awards?'):
        st.image(best_award_book)
with st.beta_expander('Is there any correlation between numbers of pages and numbers of ratings?'):
        st.image(pages_rating)
        st.write("Observing the scatter plot we can see that it looks like there is a little negative correlation. The correlation coefficient, however, is -0.04, meaning that this correlation is not statistically significant")
with st.beta_expander('Is there any difference between the normalized ratings and the mean ratings?'):
        st.image(norm_rating_dists)
        st.write("Both the normalized ratings and the mean ratings follow a normal distribution. This means we do not have many book that are rated really good or really bad. Based on this information, it seems like people tend not to give extremely high or extremely low ratings.")
with st.beta_expander('Are we sure the normal distribution is the best fit for these datas?'):
        st.image(best_fit)
        st.write('We used an algorithm to find the best possible fit, and the results above confirm that the normal distribution is actually the best fit.')
with st.beta_expander('Does the awards number follow any kind of distribution?'):
        st.image(book_awards)
        st.write('We can see that we have almost an exponential decrease of books by the increase of the number of awards')
with st.beta_expander('Is there any correlation between the ratings on goodreads and the awards won by the book?'):
        st.image(ratings_awards)
        st.write("The correlation is not significant, with a correlation coefficient of 0.03. Ratings are given by the internet people, while awards are usually given by professionals. Having no correlation between those two things show that really often the value peoples give to a book is different than the professional judgement. Population of raters from internet people is much higher than the professionals giving awards, therefore it is more of a democratic and real outlook on how good the book is.")
with st.beta_expander('What is the probability that a book that is part of a series has won an award?'):
        st.write('We can answer this question by using the Bayes Theorem.')
        st.code("""p(award|series) = [p(series|award)*p(award)] / p(series)  --> Bayes Theorem

        prob_series_award = 214/500

        prob_award = 550 / (550+446)

        prob_series = 351 / (351+645)

        prob_award_series = (prob_series_award * prob_award) / prob_series
        
        p(series|award) =  0.428
        p(award) =  0.5522088353413654
        p(series) =  0.35240963855421686

        Therefore,
        p(award|series) =  0.6706552706552706)""")
        st.write("The probability that a book that is part of a series has won an award is 67%, so it is really likely if a book has a series, it will also get an award")
with st.beta_expander("How likely is a book to win an award based on his genre?"):
        st.image(mean_awards_genres)
        st.write("Science Fiction is the genre with most awards.")
with st.beta_expander("How many awards did every genre won during the 20th century?"):
        st.image(total_awards_genres)
st.text(' ')
st.text(' ')

# function description
st.header('How does this app work? :hammer_and_wrench:')
st.write('If you are curious about how this app works, check the dropdown options below!')

with st.beta_expander('How does the column selection work?'):
        st.write('The column selection tool work with a multiselect box method of Streamlit, combined with Pandas.')
        st.code("""columns_to_show = st.multiselect("Select the columns you want to display", df.columns)
st.dataframe(df[columns_to_show])""", language='python')
        st.write("We store the result of the multiselction in a variable and we call that variable as columns index when showing the dataframe.")

with st.beta_expander('How do the filtering functions work?'):
        st.write("The filtering functions work with input method of Streamlit, and uses that input to filter the dataframe with Pandas")
        st.code("""min_rating = st.number_input("Minimum Rating (from 0 to 10)", min_value=0, max_value=10)
max_rating = st.number_input("Maximum Rating (from 0 to 10)", min_value=0, max_value=10, value=10)

min_year = st.number_input("Minimum Year (from 1900 to 2003)", min_value=1900)
max_year = st.number_input("Maximum Year (from 1900 to 2003)", min_value=1900, max_value=2003, value=2003)

min_pag = st.number_input("Minimum Pages (from 23 to 14777)", min_value=23)
max_pag = st.number_input("Maximum Pages (from 23 to 14777)", min_value=0, max_value=14777, value=14777)

min_award = st.number_input("Minimum Awards (from 0 to 28)", min_value=0)
max_award = st.number_input("Maximum Awards (from 0 to 28)", min_value=0, max_value=28, value=28)

st.dataframe(df.query("@min_rating<=minmax_norm_ratings<=@max_rating & @min_year<=pub_year<=@max_year & @min_pag<=N_pag<=@max_pag & @min_award<=Awards<=@max_award"))""", language='python')
        st.write("We store the result of every input in a variable, then we use those variable to query the dataframe")

with st.beta_expander('How does the search function work?'):
        st.write('The search function work with input method of Streamlit, combined with Pandas')
        st.code("""author_input = st.text_input('Author Name', 'Author Name')
st.dataframe(df[df['Author'].str.contains(author_input)])""")
        st.write('We store the result of the input in a variable. To make the function works even with incomplete inputs, we used the str.contains method of Pandas')



with st.beta_expander("How did we get our data ? (The Scraper)"):
                st.write("""The technique that we have used to extract our data is Web Scraping. The website that we target to get our data is
                [GoodReads](https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century) which is a website that contains a database of all the known books.
                We have focused ourself on the best books of 20th Century.""")

                st.write("""To do that we've write a software in python using the large quantity of library that it has. 
                [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) is the python package that we have used to that, BS permits us to 
                parse HTML and XML documents""")
                st.code("""from bs4 import BeautifulSoup""")
                st.write("""To use Beautiful Soup we have to define the method, and pass to it the section of the HTML code to get and parse it, but before doing this we
                have to initialize BS by passing it a request object that contain a link. The main issue that the we had during this process is that we had to scrape our data from a lot of web pages
                ,but the requests package was really slow doing that, so after a lot of research we have find out that the best solution to speed up the loadings was to paralelize the websites requests.
                To do that we have found the [Grequests Package](https://github.com/spyoungtech/grequests), an Asynchronous Requests library, and we have implemented it on our scraper.""")
                st.code("""
                import grequests
                def grequest_page(strings, index):
                reqs = (grequests.get(string) for string in strings)
                resp = grequests.imap(reqs, grequests.Pool(index))
                return resp
                """)
                st.write("""Once solved the loading time issue, we had to get the data for a thousands books, so by parsing the base url wich is the link to the pages where the books are 
                listed, with our scraper we targeted the url of each book and we store it in a list and in a file.txt locally as well as redundacy.""")
                st.code("""
                #BASE URL, WHERE ALL THE BOOKS WHERE LISTED, 100 BOOKS ON EACH PAGE
                link_base_string = "https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century?page={}" 
                #BY CALLING THIS METHOD YOU GET THE LINK OF THE BOOOK
                info_book = bs.find('a', class_ = 'bookTitle')
                book_link = "https://www.goodreads.com{}".format(info_book['href'])
                """)
                st.write("Piece of the html code of the page scraped, where to get url by parsing the href item.")
                st.code("""<a class="bookTitle" itemprop="url" href="/book/show/2657.To_Kill_a_Mockingbird">
                <span itemprop="name" role="heading" aria-level="4">To Kill a Mockingbird</span>
                </a>""")
                st.write("""Once we had the links we written a function to scrape the data of the books, by parsing each book individualy.
                The data that we wanted were: 
                - Title (string)
                - Author (string)
                - Rating Counter (int)
                - Review Counter (int)
                - Rating Value (float)
                - Number of the Pages (int)
                - The first pubblication Year (string)
                - If it's part of a series or not (boolean value)
                - The first 3 Genres (list)
                - How many awards (int)
                - Locations where the plot was based
                Below we've left some code snippets of the most difficult items to scrape and filter.""")
                st.code("""
                #FUNCTION TO GET 3 GENRES
                def three_genres(book):
                        genres = []
                        names = book.find_all('a', class_="actionLinkLite bookPageGenreLink")
                        if names is not None:
                                for name in names:
                                        genres.append(name.get_text())
                                        genres = genres[:3]  
                                return genres
                        else:
                                return np.nan
                #FUNCTION TO GET THE AWARDS
                def get_awards(book):
                        awards_list = []    
                        names = book.find_all('a', class_="award")
                        if names is not None:
                                for name in names:
                                        awards_list.append(name.get_text())
                                return awards_list
                        else: 
                                return np.nan
                """)
                st.write("""After getting all the data we stored them into a dictionary and added into a dataframe.
                With a for loop of 1000 cycles and a lot of patience we exported the dataFrame into a csv file.
                For some more info about the code have a look to the 2 versions on our repo:
                - [Good_Reads_Scraper 1.1](https://github.com/lorenzodemiri/Project_Snape/blob/kimberley/goodreads_short_scraper.py)
                - [Good_Reads_Scraper 1.0](https://github.com/lorenzodemiri/Project_Snape/blob/kimberley/goodreads_scraper.py)""")

st.header('Technologies :floppy_disk:')
st.write("Project is created with:\n")

st.image(python_image)
st.image(pandas_image)
st.image(streamlit_image)
st.image(matplotlib_image)
st.image(numpy_image)
st.image(seaborn_image)
