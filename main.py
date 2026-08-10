import logging

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI

import rag_core
from custom_types import RAGUpsertResult, RAGQueryResult

load_dotenv()

logger = logging.getLogger("uvicorn")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)

    def _ingest() -> RAGUpsertResult:
        return rag_core.ingest_pdf(pdf_path, source_id)

    result = await ctx.step.run(
        "load-embed-upsert",
        _ingest,
        output_type=RAGUpsertResult,
    )
    return result.model_dump()


@inngest_client.create_function(
    fn_id="rag-query-pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    def _answer() -> RAGQueryResult:
        return rag_core.answer_question(question, top_k)

    result = await ctx.step.run(
        "retrieve-and-answer",
        _answer,
        output_type=RAGQueryResult,
    )
    return result.model_dump()


app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai],
)
