import os
import re
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_together import Together
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# ========== CẤU HÌNH ==========
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

# Tải mô hình
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")
vector_store = FAISS.load_local(
    STORAGE_DIR / "vector_store", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Khởi tạo LLM
llm = Together(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.2,
    max_tokens=1024,
    top_k=3
)

# Tải dữ liệu FAQs
with open(STORAGE_DIR / "faqs.pkl", "rb") as f:
    faqs = pickle.load(f)

# Tải BM25 indexes
with open(STORAGE_DIR / "bm25_keywords.pkl", "rb") as f:
    bm25_keywords = pickle.load(f)

with open(STORAGE_DIR / "bm25_questions.pkl", "rb") as f:
    bm25_questions = pickle.load(f)

# ========== TIỆN ÍCH ==========
def print_colored(text, color):
    """In màu cho terminal"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

# ========== HÀM CHỨC NĂNG CHÍNH ==========
def extract_keywords(question: str) -> list:
    """Trích xuất từ khóa từ câu hỏi dùng LLM"""
    prompt = f"""
    BẠN LÀ HỆ THỐNG TRÍCH XUẤT TỪ KHÓA. Hãy trích xuất các từ khóa quan trọng từ câu hỏi sau.
    CHỈ TRẢ LỜI BẰNG DANH SÁCH TỪ KHÓA PHÂN CÁCH BẰNG DẤU PHẨY.
    
    CÂU HỎI: {question}
    TỪ KHÓA:
    """
    response = llm.invoke(prompt).strip()
    
    # Làm sạch kết quả
    keywords = []
    for kw in re.split(r'[,;|.\n]+', response):
        clean_kw = kw.strip()
        if clean_kw and len(clean_kw) > 1:  # Loại bỏ từ đơn lẻ
            keywords.append(clean_kw.lower())
    
    print_colored(f"\n🔑 Từ khóa trích xuất: {', '.join(keywords)}", "yellow")
    return keywords

def hybrid_search(question: str, keywords: list, top_k: int = 5) -> dict:
    """
    Thực hiện hybrid search:
    - 30% keyword search (FAQs)
    - 70% vector search (Markdown)
    """
    # ===== 1. TÌM KIẾM KEYWORD-BASED (FAQs) =====
    # Tìm bằng keywords
    tokenized_keywords = " ".join(keywords).split()
    keyword_scores = bm25_keywords.get_scores(tokenized_keywords)
    
    # Tìm bằng câu hỏi tương tự
    tokenized_question = question.lower().split()
    question_scores = bm25_questions.get_scores(tokenized_question)
    
    # Kết hợp điểm số
    combined_scores = (np.array(keyword_scores) * 0.6 + (np.array(question_scores) * 0.4))
    
    # Lấy top kết quả
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    faq_results = []
    for idx in top_indices:
        if combined_scores[idx] > 0:
            faq = faqs[idx]
            faq_results.append({
                "question": faq["question"],
                "answer": faq["answer"],
                "keywords": ", ".join(faq["keywords"]),
                "score": float(combined_scores[idx])
            })
    
    # ===== 2. TÌM KIẾM VECTOR-BASED (MARKDOWN) =====
    vector_results = []
    docs = vector_store.similarity_search_with_score(question, k=top_k)
    for doc, score in docs:
        content = doc.page_content
        # Cắt ngắn nội dung để hiển thị
        if len(content) > 500:
            content = content[:250] + " ... " + content[-250:]
            
        vector_results.append({
            "source": doc.metadata.get("source", "Unknown"),
            "content": content,
            "score": float(score)
        })
    
    # ===== 3. KẾT HỢP KẾT QUẢ =====
    # Chuẩn hóa điểm số
    max_faq_score = max([r["score"] for r in faq_results], default=1)
    max_vector_score = max([r["score"] for r in vector_results], default=1)
    
    combined_results = []
    for res in faq_results:
        combined_results.append({
            "type": "FAQ",
            "content": f"Q: {res['question']}\nA: {res['answer']}",
            "keywords": res["keywords"],
            "adjusted_score": (res["score"] / max_faq_score) * 0.3 if max_faq_score > 0 else 0
        })
    
    for res in vector_results:
        combined_results.append({
            "type": "MARKDOWN",
            "source": res["source"],
            "content": res["content"],
            "adjusted_score": (res["score"] / max_vector_score) * 0.7 if max_vector_score > 0 else 0
        })
    
    # Sắp xếp theo điểm tổng hợp
    combined_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
    
    return {
        "faq_results": faq_results,
        "vector_results": vector_results,
        "combined_results": combined_results
    }

def generate_answer(question: str, context: str) -> str:
    """Tạo câu trả lời từ ngữ cảnh sử dụng LLM"""
    template = """
    BẠN LÀ TRỢ LÝ ẢO CHÍNH THỨC CỦA ĐẠI HỌC BÁCH KHOA THÀNH PHỐ HỒ CHÍ MINH (HCMUT).

## NGUYÊN TẮC HOẠT ĐỘNG CỐT LÕI:
1. **Ngôn ngữ và phong cách**: Luôn trả lời bằng tiếng Việt với giọng điệu trang trọng, chuyên nghiệp nhưng thân thiện
2. **Độ chính xác**: Chỉ cung cấp thông tin có trong cơ sở dữ liệu được cung cấp
3. **Tính súc tích**: Câu trả lời ngắn gọn, có cấu trúc rõ ràng, đi thẳng vào vấn đề
4. **Phạm vi hỗ trợ**: Chỉ trả lời các câu hỏi liên quan đến HCMUT

## CÁCH XỬ LÝ CÂU HỎI:

### Câu hỏi liên quan đến HCMUT:
- **Có thông tin**: Trả lời trực tiếp dựa trên dữ liệu
- **Không có thông tin**: "Tôi không có thông tin về vấn đề này trong cơ sở dữ liệu hiện tại. Quý phụ huynh/ học sinh/ sinh viên có thể liên hệ trực tiếp với các phòng ban liên quan của trường để được hỗ trợ."

### Câu hỏi ngoài phạm vi:
"Tôi chỉ hỗ trợ các câu hỏi liên quan đến Đại học Bách Khoa TPHCM. Quý phụ huynh/ học sinh/ sinh viên có thể hỏi về:
- Thông tin tuyển sinh và chương trình đào tạo
- Hoạt động nghiên cứu và hợp tác
- Dịch vụ sinh viên và cơ sở vật chất
- Các sự kiện và tin tức của trường"

## HƯỚNG DẪN TRẢ LỜI:
- Sử dụng cấu trúc: **Câu trả lời chính** → **Thông tin bổ sung** (nếu có) → **Hướng dẫn tiếp theo** (nếu cần)
- Với câu hỏi phức tạp: Chia nhỏ thành các điểm rõ ràng
- Luôn kết thúc bằng cách hỏi có cần hỗ trợ thêm gì khác không
- Tránh sử dụng từ ngữ mơ hồ hoặc không rõ ràng
- Không sử dụng từ ngữ quá kỹ thuật hoặc học thuật trừ khi cần thiết
- Không đưa ra ý kiến cá nhân hoặc suy đoán
-  Không sử dụng từ ngữ quá thân mật hoặc không phù hợp với môi trường học thuật


## THÔNG TIN THAM KHẢO:
{context}

## CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

## CÂU TRẢ LỜI:
    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(context=context, question=question)
    
    return llm.invoke(formatted_prompt)

# ========== GIAO DIỆN CHÍNH ==========
def main():
    print_colored("\n" + "=" * 50, "cyan")
    print_colored("HỆ THỐNG CHATBOT TRƯỜNG ĐẠI HỌC BÁCH KHOA TPHCM", "cyan")
    print_colored("=" * 50, "cyan")
    
    while True:
        try:
            question = input("\n🤖 Bạn: ")
            if question.lower() in ["exit", "quit", "thoát"]:
                break
                
            # Bước 1: Trích xuất từ khóa
            keywords = extract_keywords(question)
            
            # Bước 2: Hybrid search
            results = hybrid_search(question, keywords)
            
            # In kết quả chi tiết
            print_colored("\n=== KẾT QUẢ TÌM KIẾM FAQs ===", "green")
            for i, res in enumerate(results["faq_results"]):
                print_colored(f"\nFAQ #{i+1} (Score: {res['score']:.4f})", "yellow")
                print(f"Q: {res['question']}")
                print(f"A: {res['answer'][:150]}...")
                print(f"Keywords: {res['keywords']}")
            
            print_colored("\n=== KẾT QUẢ TÌM KIẾM MARKDOWN ===", "green")
            for i, res in enumerate(results["vector_results"]):
                print_colored(f"\nDocument #{i+1} (Score: {res['score']:.4f})", "yellow")
                print(f"Source: {res['source']}")
                print(f"Content: {res['content']}")
            
            # Tạo ngữ cảnh từ top 5 kết quả kết hợp
            context = ""
            for i, res in enumerate(results["combined_results"][:5]):
                context += f"\n\n[{i+1}] {res['type']} (Score: {res['adjusted_score']:.4f}):\n"
                
                if res["type"] == "FAQ":
                    context += res["content"]
                else:
                    context += f"Source: {res['source']}\n{res['content']}"
            
            # Bước 3: Tạo câu trả lời
            answer = generate_answer(question, context)
            
            print_colored("\n=== CÂU TRẢ LỜI ===", "blue")
            print(answer)
            
        except Exception as e:
            print_colored(f"⚠️ Lỗi: {str(e)}", "red")

if __name__ == "__main__":
    main()