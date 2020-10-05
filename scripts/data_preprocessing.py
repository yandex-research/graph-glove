import os
import logging
import argparse
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
from glove import Corpus
from gensim import downloader
from gensim.corpora.wikicorpus import tokenize


def _process_page(page):
    assert len(page['section_titles']) == len(page['section_texts'])
    assert len(page.keys()) == 3

    rows = [page['title']]

    for section_title, section_text in zip(page['section_titles'], page['section_texts']):
        rows.append(section_title)
        rows.append(section_text)

    page_tok = tokenize('\n'.join(rows))
    return ' '.join(page_tok) + '\n'


def prepare_wiki(args):
    logging.info('Preparing wiki')
    wiki = downloader.load(args.data_name)
    with Pool() as pool, open(args.data_path, 'w') as f:
        for page in tqdm(pool.imap_unordered(_process_page, wiki, chunksize=args.chunksize)):
            f.write(page)


def prepare_corpus(args):
    logging.info('Preparing corpus')
    word_counts = Counter()
    for tokens in map(str.split, open(args.data_path)):
        word_counts.update(tokens)
    logging.info('Counted {} unique words.'.format(len(word_counts)))
    logging.info('Truncating vocabulary at min_count {}, max_tokens {}'.format(args.min_count, args.max_tokens))
    tokens = {token for token, count in word_counts.most_common(args.max_tokens)
              if count >= args.min_count}
    dictionary = {token: i for i, token in enumerate(tokens)}
    logging.info('Using vocabulary of size {}'.format(len(dictionary)))
    corpus = Corpus(dictionary)
    logging.info('Counting co-occurrences. Window size {}'.format(args.window))
    corpus.fit(map(str.split, open(args.data_path)), window=args.window, ignore_missing=True)
    corpus.save(args.co_path)
    return corpus


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', default='wiki-english-20171001', type=str)
    parser.add_argument('--data-path', default='./word_embeddings/wiki_tokenized.txt', type=str)
    parser.add_argument('--co-path', default='./word_embeddings/co-occurrences.pkl', type=str)
    parser.add_argument('--window', default=10, type=int)
    parser.add_argument('--min_count', default=100, type=int)
    parser.add_argument('--max_tokens', default=50_000, type=int)
    parser.add_argument('--chunksize', default=1000, type=int)
    parsed_args = parser.parse_args()

    prepare_wiki(parsed_args)
    corp = prepare_corpus(parsed_args)
