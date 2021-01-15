# importing libraries
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading imgs
snape = Image.open("snape.png")
library = Image.open('library.jpg')
best_award_book = Image.open('best_award_books.jpg')
best_rated_book = Image.open('best_rated_books.jpg')
pages_rating = Image.open('pages_ratings.jpg')
norm_rating_dists = Image.open('norm_ratings_dists.jpg')
best_fit = Image.open('MeanNormRatings_BestFit.jpg')
book_awards = Image.open('bookawards.jpg')
ratings_awards = Image.open('ratings_awards.jpg')
mean_awards_genres = Image.open('MeanAwardsByGenre.jpg')
total_awards_genres = Image.open('BarChartGenres_Awards.jpg')

# loading dataframe
df = pd.read_csv('Books.csv')
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
#st.sidebar.markdown("""Team Snape is composed by:\n
#                    Kimberley\n
#                    Bence\n
#                    Lorenzo\n
#                    Luca\n
#                    Zakariya""")
st.sidebar.markdown("The repository for the project can be found [here](https://github.com/lorenzodemiri/Project_Snape)!")

# setting header
st.title("Analysis of the 20th Century best books")
st.image(library)

# presentation of work and data
st.header('How did we developed the project :hammer_and_wrench:')
st.write("""We took [GoodReads](https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century) list of the best book of the 20th century as our data source,
            created a program to scrape all the information we needed from there and saved it to a dataframe.""")
if st.button('Click here to see the DataFrame'):
        st.dataframe(df)
st.write("""Straight from the site we got the title, author, rating count, review count, rating value, number of pages, year of the first publication,
        if the book belongs to a series or not, genres, and how many awards the book won.""")
st.write("""Based on those datas, we decided to do a normalization of the rating and of the mean rating, scaling them from 0 to 10 to have a better understanding on how the rating distributes
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
with st.beta_expander('Search by author'):
        author_input = st.text_input('Author Name', 'Author Name')
        st.dataframe(df[df['Author'].str.contains(author_input)])
with st.beta_expander('Or by title'):
        title_input = st.text_input('Title', 'Title')
        st.dataframe(df[df['Title'].str.contains(title_input)])

# visualization
st.header('Visualizing the data :bar_chart:')
st.write("There are a lot of question we can ask about the best books of the 20th Century. We tried to answer to some of them")
with st.beta_expander('Which are the best rated books?'):
        st.image(best_rated_book)
with st.beta_expander('Which are the books that won more awards?'):
        st.image(best_award_book)
with st.beta_expander('Is there any correlation between number of pages and number of ratings?'):
        st.image(pages_rating)
        st.write("Observing the scatter plot we can see what it looks like a little negative correlation. The correlation coefficient, however, is -0.04, meaning that this correlation is not statistically significant")
with st.beta_expander('Is there any difference between the normalized ratings and the mean ratings?'):
        st.image(norm_rating_dists)
        st.write("Both the normalized ratings and the mean ratings follow a normal distribution. This mean we have not much book that are rated really good or really bad. It's fair to think if this follow the real value of the book or if people tend to not give extreme ratings")
with st.beta_expander('Are we sure the normal distribution is the best fit for these datas?'):
        st.image(best_fit)
        st.write('We used an algorithm to find the best possible fit, and the results above confirms that the normal distribution is actually the best fit.')
with st.beta_expander('Does the awards number follow any kiind of distribution?'):
        st.image(book_awards)
        st.write('We can see that we have almost an exponential decrease of books going up with the number of awards')
with st.beta_expander('Is there any correlation between the ratings on goodreads and the awards winned by the book?'):
        st.image(ratings_awards)
        st.write("The correlation is not significant, with a correlation coefficient of 0.03. Ratings are given by the internet people, while awards are usually given by professional. Having no correlation between those two things show that really often the value peoples give to a book is different than the professional judgement. Population of raters from internet people is much higher than the professionals giving awards, therefore it is more of a democratic and real outlook on how good the book is.")
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


#SCRAPING SECTION LORENZO 
def scraping_section():
        #IMAGE
        data_gif = Image.open('giphy.gif')
        
        with st.beta_expander("How did we get our data ? (The Scraper)"):
                st.write("""The technique that we have used to extract our data is Web Scraping. The website that we target to get our data is
                [GoodReads](https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century) which is a website that contains a database of all the known books.
                We have focused ourself on the best books of 20th Century.""")
                st.image(data_gif)
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