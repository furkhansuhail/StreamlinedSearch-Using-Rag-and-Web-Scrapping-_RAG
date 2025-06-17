# ====================== SAFELY PATCH TORCH.CLASSES CRASH ======================
import streamlit.watcher.local_sources_watcher as lsw

original_get_module_paths = lsw.get_module_paths

def safe_get_module_paths(module):
    name = getattr(module, "__name__", "")
    if name.startswith("torch.classes"):
        return []  # Skip modules like torch.classes.x
    try:
        return original_get_module_paths(module)
    except Exception:
        return []

lsw.get_module_paths = safe_get_module_paths
# ==============================================================================




from ImportsForModel import *
import os
import streamlit as st
from Web_RAG_Model import *
from Pdf_RAG_Model import *
from PDF_Web_RAG_Model import *
from LLM_Module import *
from LLM_Pdf_Web_RAG import *



def shutdown_streamlit():
    st.warning("Shutting down the Streamlit app...")
    os._exit(0)

def run_streamlit_app():
    file_name = ""
    st.set_page_config(page_title="Hybrid RAG Search", layout="wide")
    st.title("[Google + PDF] Semantic RAG with Google Gemma")

    topic = st.text_input("Enter your query:")
    verbose = st.toggle("Verbose Answer", value=False)
    number_results = st.slider("Number of URLs to search", 3, 20, 5)
    snippet_length = st.slider("Text snippet length (characters)", 600, 6000, 3000)
    uploaded_file = st.file_uploader("Upload PDF for knowledge base", type="pdf")
    use_full_pdf = st.checkbox("Use entire PDF (skip Chapter 1 detection)", value=False)

    if uploaded_file is not None:
        file_name = uploaded_file.name

    search_mode = st.selectbox(
        "Choose Search Mode",
        ["Web_search_RAG", "PDF_Search_RAG", "Pdf_Google_search", "LLM_GoogleResults_PDF", "LLM"]
    )

    rag_search_type = st.selectbox(
        "Choose RAG Search Method",
        ["dot product", "cosine", "euclidean", "faiss", "hybrid_BM25_Embeddings", "cross_encoder"]
    )

    search_mode_clean = search_mode.lower().strip()
    requires_pdf = "pdf" in search_mode_clean
    disable_run = requires_pdf and not uploaded_file

    run_button = st.button("Run Query", disabled=disable_run)

    if disable_run:
        st.warning("📄 This search mode requires a PDF. Please upload a PDF file to proceed.")

    try:
        if run_button and (topic or uploaded_file):
            pdf_bytes = uploaded_file.read() if (uploaded_file and requires_pdf) else None

            with st.spinner("Running pipeline..."):
                results, search_method = None, None

                if search_mode_clean == 'web_search_rag':
                    websearch_Rag_app = WEB_RAG_Application(
                        topic=topic,
                        number_results=number_results,
                        mode=search_mode,
                        pdf_bytes=pdf_bytes,
                        verbose=verbose
                    )
                    websearch_Rag_app.run_web_pipeline()
                    results, search_method = websearch_Rag_app.Semantic_RAG_Search(topic, rag_search_type)

                elif search_mode_clean == "pdf_search_rag" and pdf_bytes:
                    Ragsearch_Pdf_app = RAG_PDF_Application(
                        topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
                        verbose=verbose, filename=file_name)

                    Ragsearch_Pdf_app.run_pdf_pipeline()
                    results, search_method = Ragsearch_Pdf_app.Semantic_Rag_DotProduct_Search(topic, rag_search_type)
                    print(results)

                elif search_mode_clean == "pdf_google_search" and pdf_bytes:
                    WebApp_PDF_RAGAPP = WEB_PDF_RAG_Application(
                        topic=topic or "N/A",
                        number_results=number_results,
                        mode=search_mode,
                        pdf_bytes=pdf_bytes,
                        verbose=verbose,
                        rag_search_type=rag_search_type,
                        file_name=file_name
                    )

                    results, search_method = WebApp_PDF_RAGAPP.Data_Gathering_Processing(
                        rag_search_type=rag_search_type,
                        st_container=st,
                        use_full_pdf=use_full_pdf
                    )
                elif search_mode_clean == "llm":
                    webapp_LLM = LLM_Application(
                        topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
                        verbose=verbose)

                    results, search_method = webapp_LLM.SearchModuleSetup_LLM()

                elif search_mode_clean == "llm_googleresults_pdf" and pdf_bytes:
                    webapp_PDF_LLM_app = WEB_PDF_LLM_RAG_Application(
                        topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
                        verbose=verbose, )
                    webapp_PDF_LLM_app.Data_Gathering_Processing()
                    webapp_PDF_LLM_app.LLM_Model_Setup()
                    # webapp_LLM_PDF_app.LLM_PDF_WEB_Query_Search(topic)
                    results = webapp_PDF_LLM_app.LLM_PDF_WEB_Query_Search(query=topic)
                    search_method = results.get("search_method", "LLM") if results else "Unknown"
                if results:
                    st.success(f"Search completed using {search_method}.")
                    st.markdown("### 🔍 Top Search Results")

                    if isinstance(results, list):
                        for i, res in enumerate(results):
                            st.markdown(f"**Result {i + 1}**")
                            score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'),
                                                                               (int, float)) else "N/A"
                            st.markdown(f"**Score:** `{score}`")
                            source = res.get("source", "Unknown Source")
                            if isinstance(source, str) and "http" in source:
                                st.markdown(f"**Source:** [{source}]({source})")
                            else:
                                st.markdown(f"**Source:** {source}")
                            snippet = res.get("text", "")[:snippet_length]
                            st.markdown("**Text Snippet:**")
                            st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
                            if verbose:
                                st.markdown("**Full Text:**")
                                st.text(res.get("text", ""))
                                st.markdown("**Metadata:**")
                                st.json(res.get("metadata", {}))
                            st.markdown("---")

                    # elif isinstance(results, dict):
                    #     for section_label, result_list in [
                    #         ("📄 Top 5 PDF Results", results.get("pdf_results", [])),
                    #         ("🌐 Top 5 Web Results", results.get("web_results", [])),
                    #         ("🔀 Top 5 Combined Results", results.get("combined_results", [])),
                    #     ]:
                    #         if result_list:
                    #             st.markdown(f"### {section_label}")
                    #             for i, res in enumerate(result_list):
                    #                 st.markdown(f"**Result {i + 1}**")
                    #                 score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'),
                    #                                                                    (int, float)) else "N/A"
                    #                 st.markdown(f"**Score:** `{score}`")
                    #                 source = res.get("source", "Unknown Source")
                    #                 if isinstance(source, str) and "http" in source:
                    #                     st.markdown(f"**Source:** [{source}]({source})")
                    #                 else:
                    #                     st.markdown(f"**Source:** {source}")
                    #                 snippet = res.get("text", "")[:snippet_length]
                    #                 st.markdown("**Text Snippet:**")
                    #                 st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
                    #                 if verbose:
                    #                     st.markdown("**Full Text:**")
                    #                     st.text(res.get("text", ""))
                    #                     st.markdown("**Metadata:**")
                    #                     st.json(res.get("metadata", {}))
                    #                 st.markdown("---")
                    #         else:
                    #             st.markdown(f"### {section_label}")
                    #             st.info("No results found.")

                    elif isinstance(results, dict):
                        if "answer" in results and "top_chunks" in results:
                            # ✅ LLM-PDF-WEB Mode result
                            st.markdown("## 🧾 Final Answer")
                            st.markdown(f"> {results['answer']}")

                            st.markdown("## 🧠 Top Supporting Chunks")
                            for i, res in enumerate(results["top_chunks"]):
                                st.markdown(f"**Result {i + 1}**")
                                score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'),
                                                                                   (int, float)) else "N/A"
                                st.markdown(f"**Score:** `{score}`")
                                source = res.get("source", "Unknown Source")
                                if isinstance(source, str) and "http" in source:
                                    st.markdown(f"**Source:** [{source}]({source})")
                                else:
                                    st.markdown(f"**Source:** {source}")
                                snippet = res.get("text", "")[:snippet_length]
                                st.markdown("**Text Snippet:**")
                                st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
                                if verbose:
                                    st.markdown("**Full Text:**")
                                    st.text(res.get("text", ""))
                                    st.markdown("**Metadata:**")
                                    st.json(res.get("metadata", {}))
                                st.markdown("---")
                        else:
                            # 🔁 Fall back to handling other dict formats (like PDF/Web split)
                            for section_label, result_list in [
                                ("📄 Top 5 PDF Results", results.get("pdf_results", [])),
                                ("🌐 Top 5 Web Results", results.get("web_results", [])),
                                ("🔀 Top 5 Combined Results", results.get("combined_results", [])),
                            ]:
                                if result_list:
                                    st.markdown(f"### {section_label}")
                                    for i, res in enumerate(result_list):
                                        st.markdown(f"**Result {i + 1}**")
                                        score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'),
                                                                                           (int, float)) else "N/A"
                                        st.markdown(f"**Score:** `{score}`")
                                        source = res.get("source", "Unknown Source")
                                        if isinstance(source, str) and "http" in source:
                                            st.markdown(f"**Source:** [{source}]({source})")
                                        else:
                                            st.markdown(f"**Source:** {source}")
                                        snippet = res.get("text", "")[:snippet_length]
                                        st.markdown("**Text Snippet:**")
                                        st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
                                        if verbose:
                                            st.markdown("**Full Text:**")
                                            st.text(res.get("text", ""))
                                            st.markdown("**Metadata:**")
                                            st.json(res.get("metadata", {}))
                                        st.markdown("---")
                                else:
                                    st.markdown(f"### {section_label}")
                                    st.info("No results found.")

                    elif isinstance(results, str) and search_method == "LLM":
                        st.markdown("### 💬 LLM Response")
                        st.markdown(f"> {results.strip()}")

                    else:
                        st.warning("❗ Results are in an unexpected format.")

                    st.markdown(
                        f"🔧 Debug: `Results type: {type(results)}, Total results: {len(results) if isinstance(results, list) else 'N/A'}`")

                else:
                    st.warning("⚠️ No results returned from the pipeline.")


                # elif search_mode_clean == "llm":
                #     webapp_LLM = LLM_Application(
                #         topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
                #         verbose=verbose)
                #
                #     results, search_method = webapp_LLM.SearchModuleSetup_LLM()


                # elif search_mode_clean == "llm_googleresults_pdf" and pdf_bytes:
                #     webapp_LLM_PDF_app = WEB_PDF_RAG_Application(
                #         topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
                #         verbose=verbose, )
                #     webapp_LLM_PDF_app.Data_Gathering_Processing()
                #     webapp_LLM_PDF_app.LLM_Model_Setup()
                #     # webapp_LLM_PDF_app.LLM_PDF_WEB_Query_Search(topic)
                #     results = webapp_LLM_PDF_app.LLM_PDF_WEB_Query_Search(query=topic)
                #     search_method = results.get("search_method", "LLM") if results else "Unknown"
            # =================== DISPLAY ======================
            # if results:
            #     st.success(f"Search completed using {search_method}.")
            #     st.markdown("### 🔍 Top Search Results")
            #
            #     if isinstance(results, list):
            #         for i, res in enumerate(results):
            #             st.markdown(f"**Result {i + 1}**")
            #
            #             score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'), (int, float)) else "N/A"
            #             st.markdown(f"**Score:** `{score}`")
            #
            #             # source = res.get("source", "Unknown Source")
            #             # if "http" in source:
            #             #     st.markdown(f"**Source:** [{source}]({source})")
            #             # else:
            #             #     st.markdown(f"**Source:** {source}")
            #             source = res.get("source", "Unknown Source")
            #             if isinstance(source, str) and "http" in source:
            #                 st.markdown(f"**Source:** [{source}]({source})")
            #             else:
            #                 st.markdown(f"**Source:** {source}")
            #
            #             if verbose:
            #                 st.markdown("**Full Text:**")
            #                 st.text(res.get("text", ""))
            #                 st.markdown("**Metadata:**")
            #                 st.json(res.get("metadata", {}))
            #             else:
            #                 snippet = res.get("text", "")[:snippet_length]
            #                 st.markdown("**Text Snippet:**")
            #                 st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
            #             st.markdown("---")
            #     elif results:
            #         st.success(f"Search completed using {search_method}.")
            #         st.markdown("### 💬 Final Answer")
            #         st.markdown(f"> {results['answer']}")
            #
            #         st.markdown("### 🔍 Top Context Chunks")
            #         for i, res in enumerate(results["top_chunks"]):
            #             st.markdown(f"**Result {i + 1}**")
            #             st.markdown(f"**Score:** `{res.get('score', 0):.4f}`")
            #
            #             source = res.get("source", "Unknown Source")
            #             if isinstance(source, str) and "http" in source:
            #                 st.markdown(f"**Source:** [{source}]({source})")
            #             else:
            #                 st.markdown(f"**Source:** {source}")
            #
            #             snippet = res.get("text", "")[:snippet_length]
            #             st.markdown("**Text Snippet:**")
            #             st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
            #
            #             if verbose:
            #                 st.markdown("**Full Chunk Text:**")
            #                 st.text(res.get("text", ""))
            #                 st.markdown("**Metadata:**")
            #                 st.json(res.get("metadata", {}))
            #
            #             st.markdown("---")
            #
            #     if isinstance(results, dict):
            #
            #         for section_label, result_list in [
            #             ("📄 Top 5 PDF Results", results.get("pdf_results", [])),
            #             ("🌐 Top 5 Web Results", results.get("web_results", [])),
            #             ("🔀 Top 5 Combined Results", results.get("combined_results", [])),
            #         ]:
            #
            #             if result_list:
            #                 st.markdown(f"### {section_label}")
            #                 for i, res in enumerate(result_list):
            #                     st.markdown(f"**Result {i + 1}**")
            #                     score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'),
            #                                                                        (int, float)) else "N/A"
            #                     st.markdown(f"**Score:** `{score}`")
            #                     source = res.get("source", "Unknown Source")
            #                     if isinstance(source, str) and "http" in source:
            #                         st.markdown(f"**Source:** [{source}]({source})")
            #                     else:
            #                         st.markdown(f"**Source:** {source}")
            #                     snippet = res.get("text", "")[:snippet_length]
            #                     st.markdown("**Text Snippet:**")
            #                     st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
            #                     if verbose:
            #                         st.markdown("**Full Text:**")
            #                         st.text(res.get("text", ""))
            #                         st.markdown("**Metadata:**")
            #                         st.json(res.get("metadata", {}))
            #                     st.markdown("---")
            #             else:
            #                 st.markdown(f"### {section_label}")
            #                 st.info("No results found.")
            #     elif isinstance(results, str) and search_method == "LLM":
            #         st.markdown("### 💬 LLM Response")
            #         st.markdown(f"> {results.strip()}")
            #     else:
            #         st.warning("❗ Results are in an unexpected format.")
            #     st.markdown(f"🔧 Debug: `Results type: {type(results)}, Total results: {len(results) if isinstance(results, list) else 'N/A'}`")
            # else:
            #     st.warning("⚠️ No results returned from the pipeline.")

    except Exception as e:
        st.error(f"Error during execution: {e}")
        st.exception(e)

    st.markdown("---")
    if st.button("Exit App"):
        shutdown_streamlit()

if __name__ == "__main__":
    run_streamlit_app()


# from ImportsForModel import *
# # from LLM_PDF import LLM_PDF_Application
# import os
# import streamlit as st
# from Web_RAG_Model import *
# from Pdf_RAG_Model import *
# # from PDF_Web_RAG import *
#
#
# def shutdown_streamlit():
#     st.warning("Shutting down the Streamlit app...")
#     os._exit(0)
#
# def run_streamlit_app():
#     file_name = ""
#     st.set_page_config(page_title="Hybrid RAG Search", layout="wide")
#     st.title("[Google + PDF] Semantic RAG with Google Gemma")
#
#     topic = st.text_input("Enter your query:")
#     verbose = st.toggle("Verbose Answer", value=False)
#     number_results = st.slider("Number of URLs to search", 3, 20, 5)
#     snippet_length = st.slider("Text snippet length (characters)", 600, 6000, 3000)
#     uploaded_file = st.file_uploader("Upload PDF for knowledge base", type="pdf")
#     use_full_pdf = st.checkbox("Use entire PDF (skip Chapter 1 detection)", value=False)
#
#     if uploaded_file is not None:
#         file_name = uploaded_file.name
#
#     search_mode = st.selectbox(
#         "Choose Search Mode",
#         ["Web_search_RAG", "PDF_Search_RAG", "Pdf_Google search", "LLM_GoogleResults_PDF", "LLM"]
#     )
#
#     rag_search_type = st.selectbox(
#         "Choose RAG Search Method",
#         ["dot product", "cosine", "euclidean", "faiss", "hybrid_BM25_Embeddings", "cross_encoder"]
#     )
#
#     search_mode_clean = search_mode.lower().strip()
#     requires_pdf = "pdf" in search_mode.lower()
#     disable_run = requires_pdf and not uploaded_file
#
#     run_button = st.button("Run Query", disabled=disable_run)
#
#     if disable_run:
#         st.warning("📄 This search mode requires a PDF. Please upload a PDF file to proceed.")
#
#     try:
#         if run_button and (topic or uploaded_file):
#             pdf_bytes = uploaded_file.read() if (uploaded_file and requires_pdf) else None
#
#             with st.spinner("Running pipeline..."):
#                 results, search_method = None, None
#
#                 if search_mode_clean == 'web_search_rag':
#                     websearch_Rag_app = WEB_RAG_Application(
#                         topic=topic,
#                         number_results=number_results,
#                         mode=search_mode,
#                         pdf_bytes=pdf_bytes,
#                         verbose=verbose
#                     )
#
#
#                     websearch_Rag_app.run_web_pipeline()
#                     # results, search_method = websearch_Rag_app.Semantic_Rag_DotProduct_Search(topic, rag_search_type)
#                     results, search_method = websearch_Rag_app.Semantic_RAG_Search(topic, rag_search_type)
#
#                 elif search_mode_clean == "pdf_search_rag" and pdf_bytes:
#                     Ragsearch_Pdf_app = RAG_PDF_Application(
#                         topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                         verbose=verbose, filename=file_name)
#
#                     Ragsearch_Pdf_app.run_pdf_pipeline()
#                     results, search_method = Ragsearch_Pdf_app.Semantic_Rag_DotProduct_Search(topic, rag_search_type)
#                     # print(results)
#
#                 # Extendable: Add other modes below (PDF, hybrid, LLM, etc.)
#                 # elif search_mode_clean == "pdf_search_rag" and pdf_bytes:
#                 #     ...
#                 # elif search_mode_clean == "llm":
#                 #     ...
#                 # elif search_mode_clean == "pdf + google search":
#                 #     ...
#                 # elif search_mode_clean == "llm_googleresults_pdf" and pdf_bytes:
#                 #     ...
#
#             # =================== DISPLAY ======================
#             if results:
#                 st.success(f"Search completed using {search_method}.")
#                 st.markdown("### 🔍 Top Search Results")
#
#                 if isinstance(results, list):
#                     for i, res in enumerate(results):
#                         st.markdown(f"**Result {i + 1}**")
#
#                         score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'), (int, float)) else "N/A"
#                         st.markdown(f"**Score:** `{score}`")
#
#                         source = res.get("source", "Unknown Source")
#                         if "http" in source:
#                             st.markdown(f"**Source:** [{source}]({source})")
#                         else:
#                             st.markdown(f"**Source:** {source}")
#
#                         if verbose:
#                             st.markdown("**Full Text:**")
#                             st.text(res.get("text", ""))
#                             st.markdown("**Metadata:**")
#                             st.json(res.get("metadata", {}))
#                         else:
#                             snippet = res.get("text", "")[:snippet_length]
#                             st.markdown("**Text Snippet:**")
#                             st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
#                         st.markdown("---")
#
#                 elif isinstance(results, dict):
#                     for label, result_list in [
#                         (" Top 5 Results from PDF", results.get("pdf_results", [])),
#                         (" Top 5 Results from Web", results.get("web_results", [])),
#                         (" Top 5 Results from Combined", results.get("combined_results", [])),
#                     ]:
#                         st.markdown(f"### {label}")
#                         if not result_list:
#                             st.info("No results found.")
#                             continue
#
#                         for i, res in enumerate(result_list):
#                             st.markdown(f"**Result {i + 1}**")
#                             score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'), (int, float)) else "N/A"
#                             st.markdown(f"**Score:** `{score}`")
#
#                             source = res.get("source", "Unknown Source")
#                             if "http" in source:
#                                 st.markdown(f"**Source:** [{source}]({source})")
#                             else:
#                                 st.markdown(f"**Source:** {source}")
#
#                             snippet = res.get("text", "")[:snippet_length]
#                             st.markdown("**Text Snippet:**")
#                             st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
#                             st.markdown("---")
#
#                 elif isinstance(results, str) and search_method == "LLM":
#                     st.markdown("### 💬 LLM Response")
#                     st.markdown(f"> {results.strip()}")
#
#                 else:
#                     st.warning("❗ Results are in an unexpected format.")
#
#                 st.markdown(f"🔧 Debug: `Results type: {type(results)}, Total results: {len(results) if isinstance(results, list) else 'N/A'}`")
#
#             else:
#                 st.warning("⚠️ No results returned from the pipeline.")
#
#     except Exception as e:
#         st.error(f"Error during execution: {e}")
#         st.exception(e)
#
#     st.markdown("---")
#     if st.button("Exit App"):
#         shutdown_streamlit()
#
# if __name__ == "__main__":
#     run_streamlit_app()


# from Web_RAG_Model import *
# # from PDF_RAG_Model import *
# # from PDF_Web_RAG_Model import *
# # from LLM import *
# # from LLM_PDF import *
# # from Test import *
# # ========== STREAMLIT RUNNER ==========
#
# def shutdown_streamlit():
#     st.warning("Shutting down the Streamlit app...")
#     os._exit(0)
# def run_streamlit_app():
#     file_name = ""
#     st.set_page_config(page_title="Hybrid RAG Search", layout="wide")
#     st.title("[Google + PDF] Semantic RAG with Google Gemma")
#
#     topic = st.text_input("Enter your query:")
#     verbose = st.toggle("Verbose Answer", value=False)
#     number_results = st.slider("Number of URLs to search", 3, 20, 5)
#     snippet_length = st.slider("Text snippet length (characters)", 600, 6000, 3000)
#     uploaded_file = st.file_uploader("Upload PDF for knowledge base", type="pdf")
#     use_full_pdf = st.checkbox("Use entire PDF (skip Chapter 1 detection)", value=False)
#
#     if uploaded_file is not None:
#         file_name = uploaded_file.name
#
#     search_mode = st.selectbox(
#         "Choose Search Mode",
#         ["Web_search_RAG", "PDF_Search_RAG", "Pdf_Google search",  "LLM_GoogleResults_PDF", "LLM"]
#     )
#
#     rag_search_type = st.selectbox(
#         "Choose RAG Search Method",
#         ["dot product", "cosine", "euclidean", "faiss", "hybrid", "cross_encoder"]
#     )
#
#     # Determine if the selected search mode requires a PDF
#     search_mode_clean = search_mode.lower().strip()
#     requires_pdf = "pdf" in search_mode.lower()
#     disable_run = requires_pdf and not uploaded_file
#
#     # Show the run button with conditional disabling
#     run_button = st.button("Run Query", disabled=disable_run)
#
#     # If required but not uploaded, show warning
#     if disable_run:
#         st.warning("📄 This search mode requires a PDF. Please upload a PDF file to proceed.")
#
#     # print(search_mode, " | ", rag_search_type)
#
#
#     try:
#         if run_button and (topic or uploaded_file):
#             pdf_bytes = uploaded_file.read() if (uploaded_file and requires_pdf) else None
#
#             with (st.spinner("Running pipeline...")):
#                 results, search_method = None, None
#
#                 if search_mode_clean == 'web_search_rag':
#                     websearch_Rag_app = WEB_RAG_Application(
#                         topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                         verbose=verbose)
#
#                     websearch_Rag_app.run_web_pipeline()
#                     results, search_method = websearch_Rag_app.Semantic_Rag_DotProduct_Search(topic, rag_search_type)
#
#
#                 # elif search_mode_clean == "pdf_search_rag" and pdf_bytes:
#                 #     Ragsearch_Pdf_app = RAG_PDF_Application(
#                 #         topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                 #         verbose=verbose)
#                 #
#                 #     Ragsearch_Pdf_app.run_pdf_pipeline()
#                 #     results, search_method = Ragsearch_Pdf_app.Semantic_Rag_DotProduct_Search(topic, rag_search_type)
#                 #     print(results)
#                 #
#                 # elif "pdf + google search" in search_mode_clean and pdf_bytes:
#                 #     webapp_web_pdf_app = WEB_PDF_RAG_Application(
#                 #         topic=topic or "N/A",
#                 #         number_results=number_results,
#                 #         mode=search_mode,
#                 #         pdf_bytes=pdf_bytes,
#                 #         verbose=verbose,
#                 #         rag_search_type=rag_search_type,
#                 #         file_name=file_name
#                 #     )
#                 #
#                 #     results, search_method = webapp_web_pdf_app.Data_Gathering_Processing(
#                 #         rag_search_type=rag_search_type,
#                 #         st_container=st,
#                 #         use_full_pdf=use_full_pdf
#                 #     )
#                 #     print(results)
#                 #
#                 #
#                 # elif search_mode_clean == "llm":
#                 #     webapp_LLM = LLM_Application(
#                 #         topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                 #         verbose=verbose)
#                 #
#                 #     results, search_method = webapp_LLM.SearchModuleSetup_LLM()
#                 #     print(results)
#                 #
#                 #
#                 # elif search_mode_clean == "llm + pdf" and pdf_bytes:
#                 #     webapp_LLM_PDF_app = LLM_PDF_Application(
#                 #         topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                 #         verbose=verbose, )
#                 #     webapp_LLM_PDF_app.run_pdf_pipeline()
#
#                     # results, search_method =
#                     # webapp_LLM_PDF_app.ModelDriver()
#                     # webapp_LLM_PDF_app.SearchTest(snippet_length)
#                     # print(results)
#                 else:
#                     pass
#
#     #
#             # ===================== DISPLAY OUTPUT ======================
#             if results:
#                 st.success(f"Search completed using {search_method}.")
#
#                 if isinstance(results, list):
#                     st.markdown("### 🔍 Top Search Results")
#
#                     for i, res in enumerate(results):
#                         st.markdown(f"**Result {i + 1}**")
#                         score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'), (int, float)) else "N/A"
#                         st.markdown(f"**Score:** `{score}`")
#
#                         source = res.get("source", "Unknown Source")
#                         if "http" in source:
#                             st.markdown(f"**Source:** [{source}]({source})")
#                         else:
#                             st.markdown(f"**Source:** {source}")
#
#                         if verbose:
#                             st.markdown("**Full Text:**")
#                             st.text(res.get("text", ""))
#                         else:
#                             snippet = res.get("text", "")[:snippet_length]
#                             st.markdown("**Text Snippet:**")
#                             st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
#                         st.markdown("---")
#
#                 elif isinstance(results, dict):
#                     for label, result_list in [
#                         (" Top 5 Results from PDF", results.get("pdf_results", [])),
#                         (" Top 5 Results from Web", results.get("web_results", [])),
#                         (" Top 5 Results from Combined", results.get("combined_results", [])),
#                     ]:
#                         st.markdown(f"### {label}")
#                         if not result_list:
#                             st.info("No results found.")
#                             continue
#
#                         for i, res in enumerate(result_list):
#                             if not isinstance(res, dict):
#                                 st.warning(f" Skipped invalid result: {res}")
#                                 continue
#
#                             score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'), (int, float)) else "N/A"
#                             st.markdown(f"**Result {i + 1}**")
#                             st.markdown(f"**Score:** `{score}`")
#
#                             source_text = res.get("source", "")
#                             if "WebLink:" in source_text and "http" in source_text:
#                                 parts = source_text.split("WebLink:")
#                                 before_link = parts[0].strip(" +")
#                                 url = parts[1].strip()
#                                 st.markdown(f"**Source:** {before_link} + [WebLink]({url})  \n`{url}`")
#                             elif "http" in source_text:
#                                 url = source_text.strip()
#                                 st.markdown(f"**Source Link:** [{url}]({url})  \n`{url}`")
#                             else:
#                                 st.markdown(f"**Source:** {source_text}")
#
#                             if verbose:
#                                 st.markdown("**Full Metadata:**")
#                                 st.json(res.get("metadata", {}))
#                                 st.markdown("**Full Chunk Text:**")
#                                 st.text(res.get("text", ""))
#                             else:
#                                 snippet = res.get("text", "")[:snippet_length]
#                                 st.markdown("**Text Snippet:**")
#                                 st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
#                             st.markdown("---")
#
#                 elif isinstance(results, str) and search_method == "LLM":
#                     # Display raw LLM output string
#                     st.markdown("### 💬 LLM Response")
#                     st.markdown(f"> {results.strip()}")
#
#     except Exception as e:
#         st.error(f"Error during execution: {e}")
#         st.exception(e)
#
#     st.markdown("---")
#     if st.button("Exit App"):
#         shutdown_streamlit()
#
# if __name__ == "__main__":
#     run_streamlit_app()
#















# def run_streamlit_app():
#     file_name = ""
#     st.set_page_config(page_title="Hybrid RAG Search", layout="wide")
#     st.title("[Google + PDF] Semantic RAG with Google Gemma")
#
#     topic = st.text_input("Enter your query:")
#     verbose = st.toggle("Verbose Answer", value=False)
#     number_results = st.slider("Number of URLs to search", 3, 20, 5)
#     snippet_length = st.slider("Text snippet length (characters)", 600, 6000, 3000)
#     uploaded_file = st.file_uploader("Upload PDF for knowledge base", type="pdf")
#     use_full_pdf = st.checkbox("Use entire PDF (skip Chapter 1 detection)", value=False)
#
#     if uploaded_file is not None:
#         file_name = uploaded_file.name
#
#     search_mode = st.selectbox(
#         "Choose Search Mode",
#         ["Web_search_RAG", "PDF_Search_RAG", "pdf + google search", "llm", "llm + pdf", "llm + google search", "llm + google search + pdf"]
#     )
#
#     rag_search_type = st.selectbox(
#         "Choose RAG Search Method",
#         ["dot product", "cosine", "euclidean", "faiss", "hybrid", "cross_encoder"]
#     )
#
#     requires_pdf = "pdf" in search_mode.lower()
#     disable_run = requires_pdf and not uploaded_file
#     run_button = st.button("Run Query", disabled=disable_run)
#
#     if disable_run:
#         st.warning("📄 Please upload a PDF file to enable this search mode.")
#
#     try:
#         if run_button and (topic or uploaded_file):
#             pdf_bytes = uploaded_file.read() if (uploaded_file and requires_pdf) else None
#
#             websearch_Rag_app = WEB_RAG_Application(topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                                      verbose=verbose)
#
#             Ragsearch_Pdf_app = RAG_PDF_Application(topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                                      verbose=verbose)
#
#             webapp_web_pdf_app = WEB_PDF_RAG_Application(
#                 topic=topic or "N/A",
#                 number_results=number_results,
#                 mode=search_mode,
#                 pdf_bytes=pdf_bytes,
#                 verbose=verbose,
#                 rag_search_type=rag_search_type,
#                 file_name=file_name
#             )
#             webapp_LLM = LLM_Application(topic=topic, number_results=number_results, mode=search_mode, pdf_bytes=pdf_bytes,
#                                      verbose=verbose)
#
#             with st.spinner("Running pipeline..."):
#                 results, search_method = None, None
#
#                 if search_mode == 'Web_search_RAG':
#                     websearch_Rag_app.run_web_pipeline()
#                     results, search_method = websearch_Rag_app.Semantic_Rag_DotProduct_Search(topic, rag_search_type)
#
#                     st.success(f" Search completed using {search_method}.")
#                     st.markdown(f"### Top Retrieved Chunks ({search_method})")
#
#                     for i, res in enumerate(results):
#                         score = f"{res['score']:.4f}" if isinstance(res['score'], (int, float)) else "N/A"
#
#                         st.markdown(f"**Result {i + 1}**")
#                         st.markdown(f"🔢 **Score:** `{score}`")
#                         st.markdown(f"🌐 **Source:** {res.get('source', 'Unknown')}")
#
#                         if verbose:
#                             st.markdown("🧾 **Full Metadata:**")
#                             st.json(res.get("metadata", {}))
#                             st.markdown("📝 **Full Chunk Text:**")
#                             st.text(res["text"])
#                         else:
#                             snippet = res.get("text", "")[:snippet_length]
#                             st.markdown("📝 **Text Snippet:**")
#                             st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
#
#                         st.markdown("---")
#
#                 elif search_mode == "PDF_Search_RAG" and pdf_bytes:
#                     Ragsearch_Pdf_app.run_pdf_pipeline()
#                     pdf_results, search_method = Ragsearch_Pdf_app.Semantic_Rag_DotProduct_Search(topic, rag_search_type)
#                     st.success(f"✅ Search completed using {search_method}.")
#                     st.markdown(f"### 🔍 Top Retrieved Chunks ({search_method})")
#
#                     for i, res in enumerate(pdf_results):
#                         score = f"{res['score']:.4f}" if isinstance(res['score'], (int, float)) else "N/A"
#                         page_number = "Page Number " + str(res.get('page_number', 'Unknown'))
#                         st.markdown(f"**Result {i + 1}**")
#                         st.markdown(f"🔢 **Score:** `{score}`")
#                         st.markdown(f"🌐 **Source:** {page_number}")
#
#                         if verbose:
#                             st.markdown("🧾 **Full Metadata:**")
#                             st.json(res.get("metadata", {}))
#                             st.markdown("📝 **Full Chunk Text:**")
#                             st.text(res["text"])
#                         else:
#                             snippet = res.get("text", "")[:snippet_length]
#                             st.markdown("📝 **Text Snippet:**")
#                             st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
#
#                         st.markdown("---")
#
#                 elif "pdf + google search" in search_mode and pdf_bytes:
#                     results, search_method = webapp_web_pdf_app.Data_Gathering_Processing(
#                         rag_search_type=rag_search_type,
#                         st_container=st,
#                         use_full_pdf=use_full_pdf
#                     )
#
#                 elif "llm" in search_mode:
#                     results, search_method = webapp_LLM.SearchModuleSetup_LLM()
#                     # results, search_method = webapp_web_pdf_app.Data_Gathering_Processing(
#                     #     rag_search_type=rag_search_type,
#                     #     st_container=st,
#                     #     use_full_pdf=use_full_pdf
#                     # )
#
#             if results and isinstance(results, dict):
#                 st.success(f"Search completed using {search_method}.")
#                 for label, result_list in [
#                     (" Top 5 Results from PDF", results.get("pdf_results", [])),
#                     (" Top 5 Results from Web", results.get("web_results", [])),
#                     (" Top 5 Results from Combined", results.get("combined_results", [])),
#                 ]:
#                     st.markdown(f"### {label}")
#                     if not result_list:
#                         st.info("No results found.")
#                         continue
#
#                     for i, res in enumerate(result_list):
#                         if not isinstance(res, dict):
#                             st.warning(f" Skipped invalid result: {res}")
#                             continue
#
#                         score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'), (int, float)) else "N/A"
#                         st.markdown(f"**Result {i + 1}**")
#                         st.markdown(f"**Score:** `{score}`")
#
#                         source_text = res.get("source", "")
#                         if "WebLink:" in source_text and "http" in source_text:
#                             parts = source_text.split("WebLink:")
#                             before_link = parts[0].strip(" +")
#                             url = parts[1].strip()
#                             st.markdown(f"**Source:** {before_link} + [WebLink]({url})  \n`{url}`")
#                         elif "http" in source_text:
#                             url = source_text.strip()
#                             st.markdown(f"**Source Link:** [{url}]({url})  \n`{url}`")
#                         else:
#                             st.markdown(f"**Source:** {source_text}")
#
#                         if verbose:
#                             st.markdown("**Full Metadata:**")
#                             st.json(res.get("metadata", {}))
#                             st.markdown("**Full Chunk Text:**")
#                             st.text(res.get("text", ""))
#                         else:
#                             snippet = res.get("text", "")[:snippet_length]
#                             st.markdown("**Text Snippet:**")
#                             st.markdown(f"> {snippet}{'...' if len(snippet) >= snippet_length else ''}")
#                         st.markdown("---")
#
#     except Exception as e:
#         st.error(f"Error during execution: {e}")
#         st.exception(e)
#
#     st.markdown("---")
#     if st.button("Exit App"):
#         shutdown_streamlit()
#
# if __name__ == "__main__":
#     run_streamlit_app()
