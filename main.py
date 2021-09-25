import tensorflow as tf

from argparse import ArgumentParser

from utils.preprocessing import Preprocessing
from utils.model import ArticleTransformer

parser = ArgumentParser()
parser.add_argument(
    '--data',
    type=str,
    default='test.csv',
    help='Path to the .csv file from the /data directory'
)

def main(start_text):
    transformer = ArticleTransformer()
    output = transformer.run_fn(start_text)
    output = transformer.tokenizer.decode(output[0], skip_special_tokens=True)
    return output

if __name__ == "__main__":
    output = main('No humans were harmed in the making of this article.')