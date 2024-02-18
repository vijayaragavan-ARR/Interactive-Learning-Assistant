# INTERACTIVE LEARNING ASSISTANT

The Learning Assistant is an AI-powered educational tool that enables users to extract information from various sources, ask insightful questions, and receive comprehensive answers in a conversational manner. It leverages state-of-the-art language models and question-answering techniques to provide a personalized and engaging learning experience.

## FEATURES

- Web scraping of text content from a specified website and YouTube video.
- PDF text extraction using PyPDF2 and OCR with pytesseract.
- Document embeddings using Hugging Face InstructEmbeddings.
- Question-answering with a pre-trained language model from Replicate API.
- Interactive evaluation of AI-generated answers.

## REQUIREMENTS

- Python 3.x
- Required Python packages: langchain, PyPDF2, pytesseract, pdf2image

## SETUP
1. Clone the Repository
   ```shell
   git clone "https://github.com/vijayaragavan-ARR/Interactive-Learning-Assistant"
   ```
2. Install the required packages:

   ```shell
   pip install -r requirements.txt
   ```
3. Set up Replicate API token:
   ```python
   REPLICATE_API_TOKEN = "YOUR_REPLICATE_API_TOKEN"
   os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

## USAGE

1. **Run the Script:**
    ```shell
    python your_script_name.py
    ```

2. **Follow the Prompts:**
    - Choose whether to include website and YouTube content.
    - Input the PDF file path when prompted.

3. **Ask Questions:**
    - Interact with the system by asking questions about the provided content.
    - The AI will generate answers based on sophisticated document embeddings.

4. **Exit the Program:**
    - Type "exit," "quit," or "bye" to gracefully exit the program.

## PROGRAM
```python
# IMPORT NECESSARY LIBRARIES AND MODULES
import os
import time
import pickle
import pytesseract
from PyPDF2 import PdfReader
from langchain.llms import Replicate
from langchain.chains import LLMChain
from pdf2image import convert_from_path
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, YoutubeLoader

# SET UP REPLICATE API TOKEN
REPLICATE_API_TOKEN = "YOUR_REPLICATE_API_TOKEN"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# FUNCTION TO LOAD CONTENT FROM WEB AND YOUTUBE
def load_URL():
    website_url = input("Web URL: ")
    web_loader = WebBaseLoader(website_url)
    web_docs = web_loader.load()

    youtube_url = input("YouTube URL: ")
    yt_loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
    yt_docs = yt_loader.load()
    return web_docs + yt_docs

# FUNCTION TO EXTRACT TEXT FROM DOCUMENTS
def extract_text(docs):
    return ''.join(doc.page_content for doc in docs)

# FUNCTION TO LOAD PDF AND EXTRACT TEXT
def load_pdf(pdf_path):
    pdf=PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    if not text:
        images = convert_from_path(pdf_path)
        text = ' '.join([pytesseract.image_to_string(image) for image in images])
    return text

# FUNCTION TO COMBINE PDFs FROM A DIRECTORY
def combine_pdfs(directory_path, output_path='combined_pdf.pdf'):

    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the specified directory.")
        return

    pdf_merger = PdfMerger()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        pdf_merger.append(pdf_path)

    pdf_merger.write(output_path)
    pdf_merger.close()

    print(f"Combined PDF saved to: {output_path}")

# MAIN PDF DIRECTORY
pdf_directory_path = 'PDF'
combine_pdfs(pdf_directory_path)

# LOAD COMBINED PDF AND EXTRACT TEXT
pdf_path_combined = 'combined_pdf.pdf'
text = load_pdf(pdf_path_combined)

# ASK USER IF THEY WANT TO USE WEBSITE AND YOUTUBE CONTENT
webyt = input("Do you want to use website and YouTube content? (yes or no): ").capitalize()

# IF YES, LOAD CONTENT FROM WEB AND YOUTUBE
if webyt == 'Yes':
    docs=load_URL()
    text += extract_text(docs)

# SPLIT TEXT INTO CHUNKS FOR PROCESSING
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=2000,
chunk_overlap=250,
length_function=len)
chunks=text_splitter.split_text(text)

# SET UP VECTORSTORE FOR STORING EMBEDDINGS
store_name ="UPDATE"
if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
else:
    embeddings = HuggingFaceInstructEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

# SET UP REPLICATE MODEL
llm = Replicate( 
 model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
      model_kwargs={
          "temperature": 0.5,
          "max_length": 1000,
          "top_p": 1,
          "system_prompt":"You are an adaptive learning assistant designed to provide innovative and specific answers to learners. Your responses should be concise, avoiding unnecessary questions. Prioritize accuracy and educational value.If faced with an unclear or incoherent question, guide the user to clarity rather than providing inaccurate information. If uncertain about an answer, refrain from sharing misinformation. AND STOP AFTER ANSWERING THE QUESTION. BE SPECIFIC AND CONCISE."},
      )

# LOAD QUESTION-ANSWERING CHAIN
chain = load_qa_chain(llm=llm, chain_type="stuff")

# FUNCTION TO OPTIMIZE THE GENERATED RESPONSE
def opti_response(response):
    if '\\'in response:
        response = response.split('\\')[0]
    elif '?'in response:
        response = response.split('?')[0]
    elif '/' in response:
        response = response.split('/')[0]
    return response

# FUNCTION TO GENERATE A RESPONSE TO A USER QUERY
def ResponseQuery(query):
  if not query:
    return "I don't have a response for an empty query."
  docs = VectorStore.similarity_search(query=query, k=3)
  response = opti_response(chain.run(input_documents=docs, question=query))
  return response

# FUNCTION TO EVALUATE USER-PROVIDED ANSWERS
def evaluate_ans(ans1, ans2, query):
    eval_temp = """
    Given 2 different responses ({ans1}) and ({ans2}) to a ({query}), name the first response as AI-Answer and the second response as USER-Answer. Evaluate the responses based on the following criteria:

    1. Relevance: Assess the relevance of the content to the given context. Ensure that the response directly addresses the specified rules and requirements.
    2. Completeness: Verify if the response covers all the specified points and includes all necessary information.
    3. Grammar and Clarity: Check for grammatical errors and assess the overall clarity of the response.
    4. Improvement Suggestions: Provide constructive feedback on how the response could be improved, suggesting specific areas for enhancement or clarification.
    5. Overall Rating out of 10: Assign a numerical score to the response based on the overall quality, considering accuracy, relevance, creativity, completeness, and clarity.
    """

    evaluation_template = PromptTemplate(input_variables=["ans1", "ans2", "query"], template=eval_temp)
    evaluation_template.format(ans1=ans1, ans2=ans2, query=query)
    llm_chain = LLMChain(llm=llm, prompt=evaluation_template)
    print("AI-ANS: ",ans1,end="\n\n")
    print(llm_chain.run({"ans1": ans1, "ans2": ans2, "query": query}),end="\n\n")

#MAIN FUNCTION:
if __name__ == "__main__":
    while True:
        user_input = input("ask a question or answer a question: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Multiple choice Question: ")
            response=ResponseQuery("Ask me a multiple choice question with 4 choices:")
            print(response)
            human_ans = input(f' give me your answer for "{response}": ')
            print('Your Answer: ', human_ans)
            print('Correct Answer: ',ResponseQuery(response))
            print("Goodbye!")
            break
        if user_input.lower() == "ask":
            query = "Ask me a question."
            response = ResponseQuery(query)
            print(response)
            human_ans = input(f' give me your answer for "{response}": ')
            print('Your answer: ', human_ans)
            evaluate_ans(ResponseQuery(response), human_ans, response)
        else:
            query = input("Ask questions about your PDF file:")
            response = ResponseQuery(query)
            print("QUESTION:", query,end="\n\n")
            print("ANSWER:", response, sep=" ", end="\n\n")


```
## OUTPUT:
![output](https://github.com/vijayaragavan-ARR/Interactive-Learning-Assistant/assets/136185806/800f8c68-02e2-440a-b4de-7a50c86cbb45)



## RESULT:
This project demonstrated exceptional performance and effectiveness in delivering informative responses. The results underscore the system's adeptness in extracting information from diverse sources, culminating in coherent and context-aware answers. This affirms the learning assistant's proficiency in bridging knowledge gaps through its intelligent processing capabilities.

**Note:** 
- Make sure to replace `"YOUR_REPLICATE_API_TOKEN"` with your actual Replicate API token before running the script.
- Make sure to add pdf to the PDF Directory.

