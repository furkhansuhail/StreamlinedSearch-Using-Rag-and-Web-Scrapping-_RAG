import html
import hashlib
import requests
import json
import re
import os
import time
import textwrap
import numpy as np
import pandas as pd
import re
import spacy
import random


from langchain.schema import Document
from pathlib import Path



# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



# Data Cleanup
from langchain.schema import Document
from pathlib import Path
import requests, json, hashlib, re, html
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect  # pip install langdetect
from bs4 import BeautifulSoup
from spacy.lang.en import English


from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain.schema import Document  # ✅ Correct LangChain Document import
from readability import Document as ReadabilityDocument  # ✅ Avoid name conflict
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import nltk
from nltk.tokenize import sent_tokenize  # You must have nltk downloaded


# Importing Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available



# Torch
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Imports for FAISS and searches
import faiss
from rank_bm25 import BM25Okapi

# Pickle
import pickle

# To Calculate Runtimes
from time import perf_counter as timer



from googlesearch import search



from tqdm import tqdm

import streamlit as st

from readability import Document as ReadabilityDocument
from transformers import AutoConfig





# PDF Imports
import fitz  # PyMuPDF for reading PDFs
nltk.download('punkt')
from nltk.tokenize import sent_tokenize