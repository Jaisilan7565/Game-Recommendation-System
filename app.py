import pickle
import time
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import bs4
from bs4 import BeautifulSoup
import numpy as np
import streamlit as st
from streamlit_card import card
import requests
import sklearn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Game Recommendation",
    page_icon="🎮",
    layout="wide"
)

image_url=[]
game_href=[]

def search_image_url(query):

    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run Chrome in headless mode
    chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration
    chrome_options.add_argument('--no-sandbox')  # Bypass OS security model

    driver = webdriver.Chrome(options=chrome_options)  # You can use any other browser driver as per your choice
    # driver = webdriver.Chrome()  # You can use any other browser driver as per your choice

    # Load the webpage
    # driver.get(f"https://www.google.com/search?q={query}&tbm=isch")
    driver.get(f"https://store.steampowered.com/search/?term={query}")
    
    # Wait for the page to load (you might need to adjust the waiting time based on your page loading time)
    driver.implicitly_wait(5)  # Wait for up to 10 seconds
    
    # Get the page source after dynamic content is loaded
    page_source = driver.page_source
    
    # Close the WebDriver
    driver.quit()
    
    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    a_tags=soup.find_all('a', class_="search_result_row ds_collapse_flag")

    if a_tags:
        first_href_src = a_tags[0].get('href')
        game_href.append(first_href_src)
    else:
        game_href.append("")
    
    # Find the img tag with class "myImg"
    div_tags = soup.find_all('div', class_='col search_capsule')
    for div_tag in div_tags:
    # Find all <img> tags within the current <div> tag
        img_tags = div_tag.find('img')
        
        # Process the found <img> tags as needed
        # for img_tag in img_tags:
            # Access attributes of the img_tag
        src = img_tags['src']
        if src!="":
            image_url.append(src)
            break
        else:
            image_url.append("")
            break

st.header("Welcome to Game Recommendation System! 🙌")
games_df=pd.read_pickle(r"CP2Dep/Game_df.pkl")
recommendation_df=pd.read_pickle(r"CP2Dep/Recommendation_df.pkl")

user_id=st.text_input("Type or Select User ID",int(51580))

if user_id=="":
    st.warning("User ID cannot be empty...")
else:
    try:
        user_ids = recommendation_df['user_id'].astype('category').cat.codes
        item_ids = recommendation_df['app_id'].astype('category').cat.codes

        # Get the unique user and game ids
        unique_user_ids = recommendation_df['user_id'].astype('category').cat.categories
        unique_item_ids = recommendation_df['app_id'].astype('category').cat.categories

        # create a sparse matrix
        user_game_matrix = coo_matrix((recommendation_df['hours'], (user_ids, item_ids)))

        # Fit the model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(user_game_matrix)

        tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, stop_words='english')

        tfidf_matrix = tfidf_vectorizer.fit_transform(games_df['title'])

        def get_similar_users(user_id, user_game_matrix, model_knn, n_neighbors=6):
            distances, indices = model_knn.kneighbors(user_game_matrix.getrow(user_id), n_neighbors=n_neighbors)
            similar_users = [unique_user_ids[i] for i in indices.flatten()[1:]]
            return similar_users

        # Get the unique game ids
        unique_game_ids = recommendation_df['app_id'].astype('category').cat.categories

        def get_similar_games(game_id, tfidf_matrix, n_neighbors=6):
            # Find the positional index of the game_id
            game_index = np.where(unique_game_ids == game_id)[0][0]
            
            cosine_similarities = linear_kernel(tfidf_matrix[game_index], tfidf_matrix).flatten()
            similar_indices = cosine_similarities.argsort()[:-n_neighbors:-1]
            similar_games = [games_df['title'].iloc[i] for i in similar_indices]
            return similar_games

        def recommend_games(user_id):
            similar_users = get_similar_users(user_id, user_game_matrix, model_knn)
            similar_games = []
            for user in similar_users:
                user_games = recommendation_df[recommendation_df['user_id'] == user]['app_id'].unique()
                for game_id in user_games:
                    similar_games.extend(get_similar_games(game_id, tfidf_matrix))
            unique_game_titles = list(set(similar_games))
            return unique_game_titles

        rec_games=recommend_games(int(user_id))

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0,progress_text)
        
        for i, title in enumerate(rec_games):
            search_image_url(title)

            progress_value = (i + 1) / len(rec_games)
            my_bar.progress(progress_value,progress_text)
            # time.sleep(1)
        my_bar.empty()

        num_columns = 3
        num_elements = len(rec_games)+1
        num_rows = 2

        # Iterate over the rows
        for i in range(num_rows):
        # Create a column layout for each row
            cols = st.columns(num_columns)
                
            # Iterate over the columns in the row
            for j in range(num_columns):
                # Calculate the index of the element
                idx = i * num_columns + j
                    
                # Check if the index is within the range of elements
                if idx < num_elements:
                    # Display the element in a "card-like" format
                    with cols[j]:
                        if idx < 5:
                            card(
                                title=rec_games[idx],
                                text="",
                                # image="https://assets.gqindia.com/photos/645c750df0141edcb0cc1771/16:9/w_1280,c_limit/100-best-games-hp-b.jpg",
                                image=image_url[idx],
                                url=game_href[idx]
                                )
    except:
        st.error("User not found...")

