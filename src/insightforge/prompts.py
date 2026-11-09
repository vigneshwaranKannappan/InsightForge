"""Prompt templates for the InsightForge prompt chain.

Chain: (1) condense the follow-up question using conversation memory into a
standalone question -> (2) retrieve statistics -> (3) answer strictly from them.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You rewrite follow-up questions for a business intelligence assistant. "
                "Using the conversation so far, rewrite the latest user question as a single "
                "standalone question that names every product, region, segment, metric or "
                "time period it refers to. Return only the rewritten question."
            ),
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "Follow-up question: {question}\nStandalone question:"),
    ]
)

ANSWER_SYSTEM_PROMPT = """You are InsightForge, a business intelligence analyst for a \
small electronics retailer. Currency is Indian Rupees (INR).

Rules:
1. Use ONLY the facts in the CONTEXT below. Never invent or recompute numbers; quote \
figures exactly as they appear in the context.
2. If the context does not contain the answer, say what is missing and suggest a \
question the data can answer.
3. Be concise and specific. Name the exact product, region, segment or period.
4. Structure every answer as:
   **Answer:** one or two sentences that directly answer the question.
   **Key insights:** up to 3 bullet points with supporting figures.
   **Recommendation:** one actionable business recommendation grounded in the data.

CONTEXT:
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

EXECUTIVE_SUMMARY_QUESTION = (
    "Give an executive summary of overall sales performance: monthly trend, top and "
    "bottom products and regions, customer segments and demographics, plus key "
    "insights and recommendations."
)
