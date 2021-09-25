import tensorflow as tf

from utils.model import ArticleTransformer

def main(start_text):
    transformer = ArticleTransformer()
    output = transformer.run_fn(start_text)
    output = transformer.tokenizer.decode(output[0], skip_special_tokens=True)
    return output

if __name__ == "__main__":
    output = main('Bidirectional encoder representation of transformers')
    with open('output.txt', 'w', encoding='utf8') as f:
        f.write(output)
        f.close()