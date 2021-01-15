from gevent import monkey as curious_george
curious_george.patch_all(thread=False, select=False)
import grequests
import pandas as pd
import numpy as np
import re 
import click 
from bs4 import BeautifulSoup
link_base_string = "https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century?page={}"
def three_genres(book):
    genres = []
    if book.find_all('a', class_="actionLinkLite bookPageGenreLink") is not None:
        for name in book.find_all('a', class_="actionLinkLite bookPageGenreLink"): 
            genres.append(name.get_text())
            genres = genres[:3]
        return genres
    else: return np.nan
def get_awards(book):
    awards_list = []    
    if book.find_all('a', class_="award") is not None:
        for name in book.find_all('a', class_="award"):awards_list.append(name.get_text())
        return awards_list
    else: return np.nan
def request_soup(page):
    return BeautifulSoup(page.content ,"html.parser")
def read_books_links(page_soup):
    Links_books = []
    for book in page_soup.find_all('tr'): Links_books.append("https://www.goodreads.com{}".format(book.find('a', class_ = 'bookTitle')['href']))
    return Links_books
def grequest_page(strings, index):
    return grequests.imap((grequests.get(string) for string in strings), grequests.Pool(index))
@click.command()
@click.option('--index', default=1, help='Set how many pages would you like to scrape')
def get_info_book(index):
    links = []
    links_res = []
    df = pd.DataFrame()
    for index in range(1,index + 1): links.append(link_base_string.format(index + 1))
    for r in grequest_page(links, index + 1): links_res += read_books_links(request_soup(r))
    for r, link in zip(grequest_page(links_res, index + 1), links_res):
        for data in request_soup(r).find_all('div', class_ = 'mainContentFloat'):
            Book_dict = {
                    "Title":(data.find('h1').text.strip() if data.find('h1') is not None else np.nan),
                    "Author":data.find('a', class_ = "authorName").text if data.find('a', class_ = "authorName") is not None else np.nan,
                    "Rating Count":re.sub("\D", "", data.find('meta', itemprop = "ratingCount").text) if data.find('meta', itemprop = "ratingCount") is not None else np.nan,
                    "Review Count":re.sub("\D", "", data.find('meta', itemprop = "reviewCount").text) if data.find('meta', itemprop = "reviewCount") is not None else np.nan,
                    "Rating Value":data.find('span', itemprop = "ratingValue").text.strip() if data.find('span', itemprop = "ratingValue") is not None else np.nan,
                    "N pag":re.sub("\D", "", data.find('span', itemprop = "numberOfPages").text) if data.find('span', itemprop = "numberOfPages") is not None else np.nan,
                    "1st Pub": data.find('nobr', class_ = "greyText").text.strip().replace(")","")[-5:] if data.find('nobr', class_ = "greyText") is not None else np.nan,
                    "series":1 if "Series" in data.find('div', id = "bookDataBox").text else 0,
                    "Genres":three_genres(data),
                    "Awards":get_awards(data),
                    "Link":link}    
            df = df.append(Book_dict, ignore_index=True)
            print(df)
            break
    df.to_csv('./resources/Books.csv')
if __name__ == "__main__":    
    get_info_book()