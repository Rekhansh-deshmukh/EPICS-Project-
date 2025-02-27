from src.helper import load_pdf_file, text_split, downlaod_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineConeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data="\data")
text_chunks = text_split(extracted_data)
embeddings = downlaod_hugging_face_embeddings()

pc = Pinecone(api_key="pcsk_LQzrX_Lk3V5nRLHVYgmxJHrmsBqazdhPMYmzhsYRdf1p1eTcnsmC4LU8BVVgRzyaPreNr")

index_name = "synthmind"

pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch = PineconeVectorStore.from_documents(documents=text_chunks, embedding=embedding, index_name=index_name)
