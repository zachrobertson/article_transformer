import json
import markdown
import pandas as pd
import tensorflow as tf

from bs4 import BeautifulSoup

class Preprocessing:
    def __init__(self, data):
        self.df = pd.read_csv(f'./data/{data}')
        self.data = pd.DataFrame(columns=['author', 'title', 'html'])
        self._create_dataset()
        self._tokenize_dataset()


    def _create_dataset(self):
        for index, row in self.df.iterrows():
            with open(f'./data/{row["content_location"]}', 'r') as f:
                raw = f.read()
                html = markdown.markdown(raw)
                self.data.loc[index] = (row['author'], row['title'], html)

    def _reverse_tokenized_string(self, tokenized_string):
        index_2_word = json.loads(self.tokenization_config['index_word'])
        reversed_string = ''
        for char in tokenized_string:
            reversed_string = ' '.join([reversed_string, index_2_word[str(char)]])
        return reversed_string

    def _add_training_tags(self, html_list):
        training_html = []
        for html in html_list:
            training_html.append('<s>' + html + '<|endoftext|>')
        return training_html

    def _process_html(self, html_list):
        html_text = []
        for html in html_list:
            soup = BeautifulSoup(html, features="html.parser")
            # Remove title
            for s in soup.select('h1'):
                s.extract()
            html_text.append(soup.get_text())
        return html_text
    
    def _tokenize_dataset(self):
        authors = self.data['author'].tolist()
        title = self.data['title'].tolist()
        html_text = self._process_html(self.data['html'].tolist())
        all_text = authors + title + html_text
        vectorization_layer = tf.keras.preprocessing.text.Tokenizer(
            num_words=10000,
            filters='\t',
            lower=False,
            split=' ',
            char_level=False,
            oov_token='OOV',
        )
        vectorization_layer.fit_on_texts(all_text)
        self.tokenization_config = vectorization_layer.get_config()

        vectorized_authors = vectorization_layer.texts_to_sequences(authors)
        vectorized_title = vectorization_layer.texts_to_sequences(title)
        vectorized_html = vectorization_layer.texts_to_sequences(html_text)

        # reversed_authors = [self._reverse_tokenized_string(item) for item in vectorized_authors]
        # reversed_title = [self._reverse_tokenized_string(item) for item in vectorized_title]
        # reversed_html = [self._reverse_tokenized_string(item) for item in vectorized_html]
        # print(reversed_authors)
        # print(reversed_title)
        # print(reversed_html)

        self.tokenized_data = pd.DataFrame({
            'author_tok' : vectorized_authors,
            'title_tok' : vectorized_title,
            'html_tok' : vectorized_html
        }, columns=['author_tok', 'title_tok', 'html_tok'])
        

        
