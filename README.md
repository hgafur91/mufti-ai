# mufti-ai
An Islamic AI answering tool based on classical texts
st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from utils import load_pdfs, chunk_text, create_vectorstore


st.set_page_config(page_title="MuftiAI Scholar", layout="wide")
st.title("🕌 MuftiAI – Classical Islamic Q&A")


# Load PDFs and initialize vectorstore
with st.spinner("Loading Islamic texts (Mutūn)..."):
text = load_pdfs("pdfs")
chunks = chunk_text(text)
vectordb = create_vectorstore(chunks)
retriever = vectordb.as_retriever()


qa_chain = RetrievalQA.from_chain_type(
llm=OpenAI(temperature=0),
retriever=retriever,
return_source_documents=True
)


# Input box for user's question
question = st.text_input("Ask a scholarly Islamic question (fiqh, ʿaqīdah, ḥadīth, etc.)")


if question:
with st.spinner("MuftiAI is processing the response..."):
result = qa_chain({"query": question})


st.markdown("### 📜 Answer")
st.write(result["result"])


st.markdown("### 📚 Sources")
for doc in result["source_documents"]:
st.write(doc.metadata) # Optional: enhance this with title/page extraction later
