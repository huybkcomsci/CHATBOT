import os
import re
import csv
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import nltk
import ssl
nltk.download('punkt')
nltk.download('punkt_tab')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAQS_DIR = DATA_DIR / "FAQs"
MARKDOWN_DIR = DATA_DIR / "markdown"
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True, parents=True)

# ========== XỬ LÝ FAQs (KEYWORD-BASED) ==========
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

# Tắt xác minh SSL tạm thời
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Gọi hàm tải tài nguyên
download_nltk_resources()

def process_faqs():
    all_faqs = []
    corpus_keywords = []
    corpus_questions = []
    
    csv_files = list(FAQS_DIR.glob("*.csv"))
    if not csv_files:
        print("Không tìm thấy file CSV trong thư mục FAQs")
        return 0
    
    print(f" Đang xử lý {len(csv_files)} file FAQs...")
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                # Kiểm tra header
                first_line = f.readline().strip()
                headers = [h.strip().lower() for h in first_line.split(',')]
                
                # Reset file pointer
                f.seek(0)
                
                # Đọc với header thích hợp
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    print(f"File {csv_file.name} không có header, bỏ qua")
                    continue
                    
                for row in reader:
                    # Lấy dữ liệu với key không phân biệt hoa thường
                    question = row.get('question', '') or row.get('câu hỏi', '')
                    answer = row.get('answer', '') or row.get('câu trả lời', '') or row.get('trả lời', '')
                    keywords = row.get('keywords', '') or row.get('từ khóa', '') or row.get('keyword', '')
                    
                    # Chuẩn hóa dữ liệu
                    question = question.strip()
                    answer = answer.strip()
                    keywords = keywords.strip()
                    
                    if not question or not answer:
                        continue
                        
                    # Xử lý keywords
                    keyword_list = [kw.strip() for kw in re.split(r'[,;|]+', keywords) if kw.strip()]
                    
                    record = {
                        "question": question,
                        "answer": answer,
                        "keywords": keyword_list
                    }
                    all_faqs.append(record)
                    corpus_keywords.append(" ".join(keyword_list).lower())
                    corpus_questions.append(question.lower())
                    
        except Exception as e:
            print(f"Lỗi khi xử lý file {csv_file}: {str(e)}")
            continue
    
    if not all_faqs:
        print("Không có dữ liệu FAQ nào được tạo")
        return 0
    
    # Lưu dữ liệu FAQs
    with open(STORAGE_DIR / "faqs.pkl", "wb") as f:
        pickle.dump(all_faqs, f)
    
    # Tạo index BM25 cho keywords
    if corpus_keywords:
        tokenized_corpus_keywords = [doc.split() for doc in corpus_keywords]
        bm25_keywords = BM25Okapi(tokenized_corpus_keywords)
        with open(STORAGE_DIR / "bm25_keywords.pkl", "wb") as f:
            pickle.dump(bm25_keywords, f)
    else:
        print("Không có dữ liệu keywords để tạo BM25")
    
    # Tạo index BM25 cho câu hỏi
    if corpus_questions:
        tokenized_corpus_questions = [doc.split() for doc in corpus_questions]
        bm25_questions = BM25Okapi(tokenized_corpus_questions)
        with open(STORAGE_DIR / "bm25_questions.pkl", "wb") as f:
            pickle.dump(bm25_questions, f)
    else:
        print("Không có dữ liệu câu hỏi để tạo BM25")
    
    print(f"Đã xử lý {len(all_faqs)} mục FAQ")
    return len(all_faqs)
# ========== XỬ LÝ MARKDOWN (VECTOR-BASED) ==========
def process_markdown():
    """Xử lý tất cả file Markdown trong thư mục"""
    
    # Tìm tất cả file .md
    md_files = list(MARKDOWN_DIR.glob("**/*.md"))
    if not md_files:
        print("Không tìm thấy file markdown")
        return 0
    
    print(f"Tìm thấy {len(md_files)} file markdown")
    
    all_docs = []
    error_count = 0
    
    for md_file in md_files:
        try:
            # Sử dụng UTF-8 để đọc file
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Bỏ qua file rỗng
            if not content.strip():
                print(f"File {md_file.name} rỗng, bỏ qua")
                continue
                
            # Tạo Document đơn giản
            from langchain_core.documents import Document
            doc = Document(
                page_content=content,
                metadata={"source": str(md_file)}
            )
            all_docs.append(doc)
            print(f"Đã tải: {md_file.name}")
            
        except Exception as e:
            print(f"Lỗi khi tải file {md_file.name}: {str(e)}")
            error_count += 1
    
    if not all_docs:
        print("Không có tài liệu markdown được tải")
        return 0
    
    print(f"Đã tải {len(all_docs)} tài liệu từ {len(md_files) - error_count} file markdown (lỗi: {error_count})")
    
    # Chia nhỏ tài liệu
    text_splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Chia nhỏ thành {len(split_docs)} chunks")
    
    # Nhúng và lưu vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-embedding"
    )
    
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(STORAGE_DIR / "vector_store")
    
    print(f"Đã lưu vector store")
    return len(split_docs)
# ========== HÀM CHÍNH ==========
if __name__ == "__main__":
    print("=" * 50)
    print("BẮT ĐẦU XỬ LÝ DỮ LIỆU")
    print("=" * 50)
    
    faq_count = process_faqs()
    md_count = process_markdown()
    
    print("\n" + "=" * 50)
    print("XỬ LÝ DỮ LIỆU HOÀN TẤT!")
    print(f"- Tổng số FAQs: {faq_count}")
    print(f"- Tổng số Markdown chunks: {md_count}")
    print(f"- Dữ liệu đã lưu tại: {STORAGE_DIR}")
    print("=" * 50)