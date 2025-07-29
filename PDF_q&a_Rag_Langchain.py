import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

# Core imports
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document, BaseRetriever
from langchain.callbacks import get_openai_callback

# Advanced components
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.evaluation import load_evaluator

# Monitoring and metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# OUTPUT PARSERS AND STRUCTURED RESPONSES
# =============================================================================

class QAResponse(BaseModel):
    """Structured response model for QA system"""
    answer: str = Field(description="The main answer to the question")
    confidence: float = Field(description="Confidence score between 0-1")
    sources: List[str] = Field(description="List of source references")
    reasoning: str = Field(description="Brief explanation of reasoning")
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")

class HallucinationDetection(BaseModel):
    """Model for hallucination detection results"""
    is_hallucinated: bool = Field(description="Whether the response contains hallucinations")
    confidence: float = Field(description="Confidence in hallucination detection")
    explanation: str = Field(description="Explanation of the detection")

# =============================================================================
# RETRIEVER PERFORMANCE EVALUATION
# =============================================================================

@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retriever performance"""
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    latency: float
    
class RetrieverEvaluator:
    """Evaluates retriever performance with various metrics"""
    
    def __init__(self, ground_truth_pairs: List[Tuple[str, List[str]]]):
        """
        Args:
            ground_truth_pairs: List of (question, relevant_doc_ids) pairs
        """
        self.ground_truth = ground_truth_pairs
        
    def evaluate_retriever(self, retriever: BaseRetriever, k: int = 5) -> RetrievalMetrics:
        """Comprehensive evaluation of retriever performance"""
        precisions, recalls, rrs, ndcgs, latencies = [], [], [], [], []
        
        for question, relevant_ids in self.ground_truth:
            start_time = time.time()
            retrieved_docs = retriever.get_relevant_documents(question)[:k]
            latency = time.time() - start_time
            
            retrieved_ids = [doc.metadata.get('id', '') for doc in retrieved_docs]
            
            # Precision@K
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
            
            # Recall@K  
            recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
            
            # MRR
            rr = 0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    rr = 1 / (i + 1)
                    break
            
            # NDCG (simplified)
            dcg = sum([1 / np.log2(i + 2) for i, doc_id in enumerate(retrieved_ids) 
                      if doc_id in relevant_ids])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_ids), k))])
            ndcg = dcg / idcg if idcg > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            rrs.append(rr)
            ndcgs.append(ndcg)
            latencies.append(latency)
        
        return RetrievalMetrics(
            precision_at_k=np.mean(precisions),
            recall_at_k=np.mean(recalls),
            mrr=np.mean(rrs),
            ndcg=np.mean(ndcgs),
            latency=np.mean(latencies)
        )

# =============================================================================
# HALLUCINATION DETECTION
# =============================================================================

class HallucinationDetector:
    """Detects hallucinations in QA responses"""
    
    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=HallucinationDetection)
        
        self.hallucination_prompt = PromptTemplate(
            template="""
            You are an expert at detecting hallucinations in AI responses.
            
            Context from documents: {context}
            
            Question: {question}
            
            AI Response: {response}
            
            Analyze whether the AI response contains any hallucinations (information not supported by the context).
            Consider:
            1. Are all facts in the response supported by the context?
            2. Are there any fabricated details or statistics?
            3. Does the response make claims beyond what's in the documents?
            
            {format_instructions}
            """,
            input_variables=["context", "question", "response"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def detect_hallucination(self, question: str, response: str, context: str) -> HallucinationDetection:
        """Detect if response contains hallucinations"""
        try:
            chain = self.hallucination_prompt | self.llm | self.parser
            result = chain.invoke({
                "question": question,
                "response": response,
                "context": context
            })
            return result
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return HallucinationDetection(
                is_hallucinated=False,
                confidence=0.0,
                explanation="Detection failed"
            )

# =============================================================================
# ADVANCED PDF QA SYSTEM
# =============================================================================

class AdvancedPDFQASystem:
    """Comprehensive PDF QA system with RAG, memory, and advanced features"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Vector store and retriever
        self.vectorstore = None
        self.retriever = None
        self.compressed_retriever = None
        
        # Memory system
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=10  # Remember last 10 exchanges
        )
        
        # Output parser
        self.qa_parser = PydanticOutputParser(pydantic_object=QAResponse)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.qa_parser, 
            llm=self.fast_llm
        )
        
        # Hallucination detector
        self.hallucination_detector = HallucinationDetector(self.fast_llm)
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "avg_response_time": 0,
            "hallucination_rate": 0,
            "user_satisfaction": []
        }
        
        # Enhanced prompt template
        self.qa_prompt = PromptTemplate(
            template="""
            You are an expert AI assistant that answers questions based on PDF documents.
            
            Context from documents:
            {context}
            
            Chat History:
            {chat_history}
            
            Question: {question}
            
            Instructions:
            1. Answer the question using ONLY information from the provided context
            2. If information is not in the context, clearly state "I don't have information about this in the provided documents"
            3. Provide confidence score based on how well the context supports your answer
            4. Include specific references to sources when possible
            5. Suggest relevant follow-up questions
            6. Be precise and avoid speculation
            
            {format_instructions}
            """,
            input_variables=["context", "chat_history", "question"],
            partial_variables={"format_instructions": self.qa_parser.get_format_instructions()}
        )
    
    def load_pdfs(self, pdf_paths: List[str]) -> None:
        """Load and process PDF documents using PyMuPDF"""
        logger.info(f"Loading {len(pdf_paths)} PDF documents...")
        
        all_documents = []
        for pdf_path in pdf_paths:
            try:
                # Open PDF with PyMuPDF
                doc = fitz.open(pdf_path)
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # Extract text with better formatting
                    text = page.get_text("text")
                    
                    # Skip empty pages
                    if not text.strip():
                        continue
                    
                    # Create Document object
                    document = Document(
                        page_content=text,
                        metadata={
                            "source_file": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "id": f"{os.path.basename(pdf_path)}_page_{page_num + 1}",
                            "load_time": datetime.now().isoformat(),
                            "page_width": page.rect.width,
                            "page_height": page.rect.height,
                            "char_count": len(text),
                            "word_count": len(text.split())
                        }
                    )
                    
                    all_documents.append(document)
                
                doc.close()  # Close the PDF file
                logger.info(f"Loaded {len([d for d in all_documents if d.metadata['source_file'] == os.path.basename(pdf_path)])} pages from {pdf_path}")
                
            except Exception as e:
                logger.error(f"Failed to load {pdf_path}: {e}")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(all_documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retrievers
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        
        # Contextual compression retriever
        compressor = LLMChainExtractor.from_llm(self.fast_llm)
        self.compressed_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        
        logger.info("Vector store and retrievers created successfully")
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context"""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    def answer_question(self, question: str, use_compression: bool = True) -> Dict[str, Any]:
        """Answer question with comprehensive analysis"""
        start_time = time.time()
        
        try:
            # Choose retriever
            retriever = self.compressed_retriever if use_compression else self.retriever
            
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(question)
            context = self._format_docs(docs)
            
            # Get chat history
            chat_history = self.memory.chat_memory.messages
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-6:]])
            
            # Generate response with structured output
            with get_openai_callback() as cb:
                chain = self.qa_prompt | self.llm | self.fixing_parser
                
                response = chain.invoke({
                    "context": context,
                    "chat_history": history_str,
                    "question": question
                })
            
            # Detect hallucinations
            hallucination_result = self.hallucination_detector.detect_hallucination(
                question=question,
                response=response.answer,
                context=context
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update memory
            self.memory.save_context(
                {"question": question},
                {"answer": response.answer}
            )
            
            # Update metrics
            self._update_metrics(response_time, hallucination_result.is_hallucinated)
            
            # Prepare comprehensive result
            result = {
                "answer": response.answer,
                "confidence": response.confidence,
                "sources": response.sources,
                "reasoning": response.reasoning,
                "follow_up_questions": response.follow_up_questions,
                "hallucination_detection": {
                    "is_hallucinated": hallucination_result.is_hallucinated,
                    "confidence": hallucination_result.confidence,
                    "explanation": hallucination_result.explanation
                },
                "metadata": {
                    "response_time": response_time,
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost,
                    "num_source_docs": len(docs),
                    "retriever_type": "compressed" if use_compression else "standard"
                },
                "source_documents": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "source": doc.metadata.get("source_file", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown")
                    }
                    for doc in docs
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "error": str(e)
            }
    
    def _update_metrics(self, response_time: float, is_hallucinated: bool) -> None:
        """Update system metrics"""
        self.metrics["total_queries"] += 1
        
        # Update average response time
        current_avg = self.metrics["avg_response_time"]
        total_queries = self.metrics["total_queries"]
        self.metrics["avg_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
        # Update hallucination rate
        current_hall_rate = self.metrics["hallucination_rate"]
        hallucination_count = current_hall_rate * (total_queries - 1) + (1 if is_hallucinated else 0)
        self.metrics["hallucination_rate"] = hallucination_count / total_queries
    
    def evaluate_system(self, test_questions: List[str]) -> Dict[str, Any]:
        """Comprehensive system evaluation"""
        logger.info("Starting system evaluation...")
        
        evaluation_results = {
            "total_questions": len(test_questions),
            "responses": [],
            "avg_confidence": 0,
            "avg_response_time": 0,
            "hallucination_rate": 0,
            "retriever_metrics": None
        }
        
        total_confidence = 0
        total_time = 0
        hallucination_count = 0
        
        for question in test_questions:
            result = self.answer_question(question)
            
            evaluation_results["responses"].append({
                "question": question,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "response_time": result["metadata"]["response_time"],
                "is_hallucinated": result["hallucination_detection"]["is_hallucinated"]
            })
            
            total_confidence += result["confidence"]
            total_time += result["metadata"]["response_time"]
            if result["hallucination_detection"]["is_hallucinated"]:
                hallucination_count += 1
        
        evaluation_results["avg_confidence"] = total_confidence / len(test_questions)
        evaluation_results["avg_response_time"] = total_time / len(test_questions)
        evaluation_results["hallucination_rate"] = hallucination_count / len(test_questions)
        
        return evaluation_results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        return {
            **self.metrics,
            "memory_size": len(self.memory.chat_memory.messages),
            "vectorstore_size": self.vectorstore._collection.count() if self.vectorstore else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Memory cleared")
    
    def export_conversation(self) -> List[Dict[str, str]]:
        """Export conversation history"""
        messages = self.memory.chat_memory.messages
        conversation = []
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                conversation.append({
                    "question": messages[i].content,
                    "answer": messages[i + 1].content,
                    "timestamp": datetime.now().isoformat()
                })
        
        return conversation

# =============================================================================
# STREAMLIT USER INTERFACE
# =============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Advanced PDF QA System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🧠 Advanced PDF Question-Answering System")
    st.markdown("*Powered by LangChain, RAG, PyMuPDF, and Advanced AI Techniques*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue")
            return
        
        # File upload
        st.header("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True
        )
        
        # Advanced options
        st.header("🔧 Advanced Options")
        use_compression = st.checkbox("Use Contextual Compression", value=True)
        show_sources = st.checkbox("Show Source Documents", value=True)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)
    
    # Initialize session state
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Process uploaded files
    if uploaded_files and api_key:
        if st.button("🔄 Process PDFs"):
            with st.spinner("Processing PDFs..."):
                try:
                    # Save uploaded files
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = f"temp_{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        pdf_paths.append(file_path)
                    
                    # Initialize QA system
                    qa_system = AdvancedPDFQASystem(api_key)
                    qa_system.load_pdfs(pdf_paths)
                    st.session_state.qa_system = qa_system
                    
                    # Clean up temp files
                    for path in pdf_paths:
                        os.remove(path)
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} PDF files!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {str(e)}")
    
    # Main chat interface
    if st.session_state.qa_system:
        # Display metrics
        if show_metrics:
            col1, col2, col3, col4 = st.columns(4)
            metrics = st.session_state.qa_system.get_system_metrics()
            
            with col1:
                st.metric("Total Queries", metrics["total_queries"])
            with col2:
                st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
            with col3:
                st.metric("Hallucination Rate", f"{metrics['hallucination_rate']:.1%}")
            with col4:
                st.metric("Memory Size", metrics["memory_size"])
        
        # Chat interface
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "metadata" in message:
                    st.write(message["content"])
                    
                    # Show additional information
                    with st.expander("📊 Response Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Confidence:** {message['metadata']['confidence']:.2f}")
                            st.write(f"**Response Time:** {message['metadata']['response_time']:.2f}s")
                        with col2:
                            hallucination = message['metadata']['hallucination_detection']
                            status = "⚠️ Potential Hallucination" if hallucination['is_hallucinated'] else "✅ Verified"
                            st.write(f"**Status:** {status}")
                            st.write(f"**Detection Confidence:** {hallucination['confidence']:.2f}")
                    
                    # Show sources
                    if show_sources and "source_documents" in message["metadata"]:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(message["metadata"]["source_documents"]):
                                st.write(f"**Source {i+1}:** {doc['source']} (Page {doc['page']})")
                                st.write(doc["content"])
                                st.divider()
                else:
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_system.answer_question(
                        prompt, 
                        use_compression=use_compression
                    )
                
                st.write(result["answer"])
                
                # Store complete response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": result
                })
        
        # Additional controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Memory"):
                st.session_state.qa_system.clear_memory()
                st.session_state.messages = []
                st.success("Memory cleared!")
        
        with col2:
            if st.button("📊 Run Evaluation"):
                test_questions = [
                    "What are the main topics covered in the documents?",
                    "Can you summarize the key findings?",
                    "What methodology was used?"
                ]
                
                with st.spinner("Running evaluation..."):
                    eval_results = st.session_state.qa_system.evaluate_system(test_questions)
                
                st.json(eval_results)
        
        with col3:
            conversation = st.session_state.qa_system.export_conversation()
            if conversation:
                st.download_button(
                    "💾 Export Conversation",
                    data=json.dumps(conversation, indent=2),
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()