from gevent import monkey as curious_george
curious_george.patch_all(thread=False, select=False)
import requests
import grequests
import pandas as pd
import numpy as np
import re 
from bs4 import BeautifulSoup
import time
### Starting the timer ###
start_time = time.time()

link_base_string = "https://www.goodreads.com/list/show/6.Best_Books_of_the_20th_Century?page={}"

#print(links)
def get_pages_links(string, index):
    links = []
    for index in range(1,index):
        url = string.format(index)
        links.append(url)
    return links
    
def request_page(strings, index):
    reqs = (grequests.get(string) for string in strings)
    resp = grequests.imap(reqs, grequests.Pool(index))
    return resp

def request_single_page(string):
    res = requests.get(string)
    return res

def request_soup(page):
    soup = BeautifulSoup(page.content ,"html.parser")
    return soup

def close_request(request_page):
    request_page.close()
    return

def read_bookslink(page_soup):
    Links = []
    #table = page_soup.find_all('table', class_ = 'tableList js-dataTooltip')
    #print(table)
    books = page_soup.find_all('tr')
    for book in books:
        info_book = book.find('a', class_ = 'bookTitle')
        book_link = "https://www.goodreads.com{}".format(info_book['href'])
        Links.append(book_link)    
    return Links

def get_bookslink(index):
    Links_res = []
    page = request_page(get_pages_links(link_base_string, index), index)
    for r in page: 
        soup = request_soup(r)
        Links_res = Links_res + read_bookslink(soup)
    print(len(Links_res))
    with open("Links_for_each_book.txt", "w") as output:
        output.write(str(Links_res))
    return Links_res

def get_pupDate(data):  
    if data.find('nobr', class_ = "greyText") is not None:
        firstPub = data.find('nobr', class_ = "greyText").text
        firstPub = firstPub.strip()
        firstPub = firstPub[-5:]
        firstPub = firstPub.replace(")","")
        return firstPub
    else: return np.nan

def three_genres(book):
    genres = []
    if  book.find_all('a', class_="actionLinkLite bookPageGenreLink") is not None:
        for name in book.find_all('a', class_="actionLinkLite bookPageGenreLink"):
            genres.append(name.get_text())
            genres = genres[:3]
        return genres
    else:
        return np.nan

def get_awards(book):
    awards_list = []
    if book.find_all('a', class_="award") is not None:
        for name in book.find_all('a', class_="award"):
            awards_list.append(name.get_text())
        return awards_list
    else: 
        print('NAN')
        return np.nan


def get_places(book):
    return

def get_info_book(page_soup, link, index):
    book = page_soup.find_all('div', class_ = 'mainContentFloat')
    for data in book:
        #TITLE of the Book
        if data.find('h1') is not None:
            title = data.find('h1').text
            title = title.strip()
        else: title = np.nan
        #AUTHOR of the book
        if data.find('a', class_ = "authorName") is not None:
            author = data.find('a', class_ = "authorName").text
        else: author = np.nan
        #RATING COUNT of the book
        if data.find('meta', itemprop = "ratingCount") is not None:
            ratingCount = data.find('meta', itemprop = "ratingCount").text
            ratingCount = re.sub("\D", "", ratingCount)
        else: ratingCount = np.nan
        #REVIEW COUNT of the book
        if data.find('meta', itemprop = "reviewCount") is not None:
            reviewCount = data.find('meta', itemprop = "reviewCount").text
            reviewCount = re.sub("\D", "", reviewCount)
        else: reviewCount = np.nan
        #RATING VALUE of the book
        if data.find('span', itemprop = "ratingValue") is not None:
            ratingValue = data.find('span', itemprop = "ratingValue").text
            ratingValue = ratingValue.strip()
        else: ratingValue = np.nan
        #NUMBER OF PAGES of the book
        if data.find('span', itemprop = "numberOfPages") is not None:
            numberOfPages = data.find('span', itemprop = "numberOfPages").text
            numberOfPages = re.sub("\D", "", numberOfPages)
        else: numberOfPages = np.nan
        #YEAR OF FIRST PUBBLICATION of the book
        firstPub = get_pupDate(data)
        #CHECK IF IT IS A SERIES OR NOT
        series = data.find('div', id = "bookDataBox").text
        if "Series" in series: 
            series = '1'
        else: 
            series = '0'
        #GENRES of the book
        genreList = three_genres(data)
        #AWARDS of the book
        if len(get_awards(data)) >= 1:
            awards = get_awards(data)
        else: awards = np.nan
        #PLACES of the book
        

        Book_dict = {
                "Link":link,
                "Title":title,
                "Author":author,
                "Rating Count":ratingCount,
                "Review Count":reviewCount,
                "Rating Value":ratingValue,
                "N pag":numberOfPages,
                "1st Pub":firstPub,
                "series":series,
                "Genres":genreList,
                "Awards":awards}
        #print(index,"  ",link,"\n__",title,author,ratingCount, reviewCount, ratingValue, numberOfPages,firstPub, series, genreList, awards,"\n\n")
        return Book_dict

def create_csv():
    links = get_bookslink(11)
    res_dict = {}
    i = 1
    page = request_page(links, 11)
    df = pd.DataFrame()
    for r, link in zip(page, links): 
        soup = request_soup(r)
        value = get_info_book(soup, link, i)
        if value is not None:
            df = df.append(value, ignore_index=True)
        i = i + 1
        print(df.tail())
    df.to_csv('./Books.csv')
        
create_csv()
#print(get_info_book(request_soup(request_single_page("https://www.goodreads.com/book/show/13496.A_Game_of_Thrones")),"https://www.goodreads.com/book/show/13496.A_Game_of_Thrones", 1))
#links = get_bookslink(11)



print("--- %s seconds ---" % (time.time() - start_time))